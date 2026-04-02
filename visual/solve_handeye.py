"""Solve hand-eye calibration from collected pose data (eye-to-hand).

Tries all OpenCV methods first. If all have residuals > threshold,
falls back to scipy Nelder-Mead multi-start optimization.
Saves the best result found.

Input:  visual/calib_data/poses.npz
Output: visual/calib_data/T_cam2base.npz   (use in code)
        visual/calib_data/T_cam2base.yaml  (human-readable)

Run:
  python3 visual/solve_handeye.py
"""

import pathlib
import datetime
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

_ROOT    = pathlib.Path(__file__).parent.parent
DATA_DIR = _ROOT / "visual" / "calib_data"
INPUT    = DATA_DIR / "poses.npz"
OUT_NPZ  = DATA_DIR / "T_cam2base.npz"
OUT_YAML = DATA_DIR / "T_cam2base.yaml"

OPENCV_GOOD_THRESH_MM  = 15.0   # accept OpenCV result if trans residual < this
SCIPY_N_STARTS         = 300    # number of random initialisations for scipy


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t.ravel()
    return T


def consistency_residuals(R_g2b, t_g2b, R_t2c, t_t2c, R_c2b, t_c2b):
    """
    Eye-to-hand consistency check:
      T_target2gripper_i = inv(T_gripper2base_i) @ T_cam2base @ T_target2cam_i
    Should be constant across all poses. Returns spread (std) of translation [mm]
    and rotation [deg].
    """
    T_c2b = make_T(R_c2b, t_c2b)
    t_list, r_list = [], []
    for Rg, tg, Rt, tt in zip(R_g2b, t_g2b, R_t2c, t_t2c):
        T_g2b = make_T(Rg, tg)
        T_t2c = make_T(Rt, tt)
        Y = np.linalg.inv(T_g2b) @ T_c2b @ T_t2c   # = T_target2gripper_i
        t_list.append(Y[:3, 3])
        r_list.append(Rotation.from_matrix(Y[:3, :3]).as_rotvec())
    t_arr = np.array(t_list)   # (N,3) metres
    r_arr = np.array(r_list)   # (N,3) rad
    trans_std = np.linalg.norm(t_arr.std(axis=0)) * 1000   # mm
    rot_std   = np.degrees(np.linalg.norm(r_arr.std(axis=0)))  # deg
    return trans_std, rot_std


# ---------------------------------------------------------------------------
# Scipy optimisation — minimise translation consistency across poses
# ---------------------------------------------------------------------------

def _T_from_vec(vec):
    """6-vec [rx,ry,rz, tx,ty,tz] -> 4x4 transform."""
    R = Rotation.from_rotvec(vec[:3]).as_matrix()
    t = vec[3:]
    return make_T(R, t)


def _objective(vec, T_b2g_list, T_t2c_list):
    """
    Y_i = T_base2gripper_i @ X @ T_target2cam_i  should equal T_target2gripper (constant).
    Minimise variance of both translation AND rotation across all poses.
    """
    X = _T_from_vec(vec)
    t_list = []
    r_list = []
    for T_b2g, T_t2c in zip(T_b2g_list, T_t2c_list):
        Y = T_b2g @ X @ T_t2c
        t_list.append(Y[:3, 3])
        r_list.append(Rotation.from_matrix(Y[:3, :3]).as_rotvec())
    t_arr = np.array(t_list)
    r_arr = np.array(r_list)
    return float(np.sum((t_arr - t_arr.mean(axis=0)) ** 2) +
                 np.sum((r_arr - r_arr.mean(axis=0)) ** 2))


def scipy_solve(R_g2b, t_g2b, R_t2c, t_t2c, n_starts=SCIPY_N_STARTS):
    """Multi-start Nelder-Mead optimisation for hand-eye calibration."""
    print(f"\n[scipy] Running {n_starts} random-start optimisations...")

    # Build T_base2gripper and T_target2cam lists
    T_b2g_list = [np.linalg.inv(make_T(R, t)) for R, t in zip(R_g2b, t_g2b)]
    T_t2c_list = [make_T(R, t) for R, t in zip(R_t2c, t_t2c)]

    best_cost = np.inf
    best_vec  = None

    rng = np.random.default_rng(42)
    for i in range(n_starts):
        # Random initial guess: small random rotation + random translation
        rv0 = rng.uniform(-np.pi, np.pi, 3)
        t0  = rng.uniform(-1.0, 1.0, 3)
        x0  = np.concatenate([rv0, t0])

        res = minimize(_objective, x0,
                       args=(T_b2g_list, T_t2c_list),
                       method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
        if res.fun < best_cost:
            best_cost = res.fun
            best_vec  = res.x

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_starts}] best cost so far: {best_cost:.6f}")

    X = _T_from_vec(best_vec)
    R_c2b = X[:3, :3]
    t_c2b = X[:3,  3]

    # Ensure valid rotation matrix (project onto SO(3))
    U, _, Vt = np.linalg.svd(R_c2b)
    R_c2b = U @ Vt
    if np.linalg.det(R_c2b) < 0:
        R_c2b = U @ np.diag([1, 1, -1]) @ Vt

    print(f"  scipy best cost: {best_cost:.6f}")
    return R_c2b, t_c2b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found. Run collect_calib_poses.py first.")
        return

    data = np.load(str(INPUT))
    N = int(data["n_captures"])
    print(f"Loaded {N} pose pairs from {INPUT}\n")

    R_g2b = list(data["R_gripper2base"])
    t_g2b = list(data["t_gripper2base"])
    R_t2c = list(data["R_target2cam"])
    t_t2c = list(data["t_target2cam"])

    # ------------------------------------------------------------------
    # 1. Try all OpenCV methods
    # ------------------------------------------------------------------
    METHODS = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}
    print(f"{'Method':<14} {'det(R)':>8} {'|t| mm':>8} "
          f"{'rot_std°':>10} {'trans_std mm':>14}")
    print("-" * 60)

    for name, method in METHODS.items():
        try:
            R_c2b, t_c2b = cv2.calibrateHandEye(
                R_g2b, t_g2b, R_t2c, t_t2c, method=method)
            det    = np.linalg.det(R_c2b)
            t_norm = np.linalg.norm(t_c2b) * 1000
            trans_std, rot_std = consistency_residuals(R_g2b, t_g2b, R_t2c, t_t2c, R_c2b, t_c2b)
            results[name] = dict(R=R_c2b, t=t_c2b, det=det,
                                 rot_std=rot_std, trans_std=trans_std)
            print(f"  {name:<12} {det:>8.5f} {t_norm:>8.1f} "
                  f"{rot_std:>10.4f} {trans_std:>14.2f}")
        except Exception as e:
            print(f"  {name:<12} FAILED: {e}")

    # Best OpenCV result (lowest translation std)
    if results:
        best_cv_name = min(results, key=lambda k: results[k]["trans_std"])
        best_cv = results[best_cv_name]
    else:
        best_cv = None

    # ------------------------------------------------------------------
    # 2. Decide whether to use OpenCV or fall back to scipy
    # ------------------------------------------------------------------
    use_scipy = (best_cv is None) or (best_cv["trans_std"] > OPENCV_GOOD_THRESH_MM)

    if not use_scipy:
        print(f"\n[OK] OpenCV {best_cv_name} is good enough "
              f"(trans_std={best_cv['trans_std']:.1f} mm < {OPENCV_GOOD_THRESH_MM} mm)")
        R_c2b  = best_cv["R"]
        t_c2b  = best_cv["t"]
        method = best_cv_name
    else:
        print(f"\n[INFO] All OpenCV methods have high residuals. "
              f"Falling back to scipy optimisation.")
        R_c2b, t_c2b = scipy_solve(R_g2b, t_g2b, R_t2c, t_t2c)
        method = "scipy_optimize"

    # ------------------------------------------------------------------
    # 3. Final result + validation
    # ------------------------------------------------------------------
    T_c2b = make_T(R_c2b, t_c2b)
    trans_std, rot_std = consistency_residuals(R_g2b, t_g2b, R_t2c, t_t2c, R_c2b, t_c2b)
    det    = np.linalg.det(R_c2b)
    t_norm = np.linalg.norm(t_c2b) * 1000

    print(f"\n=== Result ({method}) ===")
    print(f"T_cam2base:\n{np.round(T_c2b, 6)}")
    print(f"Camera position in base frame [mm]: {t_c2b.ravel()*1000}")

    print(f"\n=== Validation ===")
    print(f"  det(R)             = {det:.6f}  (expect 1.000)")
    print(f"  |t_cam2base|       = {t_norm:.1f} mm  (expect 300–2500 mm)")
    print(f"  T_target2gripper translation std = {trans_std:.2f} mm  (good < 15 mm)")
    print(f"  T_target2gripper rotation    std = {rot_std:.3f}°   (good < 2°)")

    warnings = []
    if abs(det - 1.0) > 0.005:
        warnings.append(f"det(R)={det:.4f} far from 1.0")
    if rot_std > 5.0:
        warnings.append(f"Rotation std {rot_std:.2f}° is high — result may be inaccurate")
    if trans_std > 20.0:
        warnings.append(f"Translation std {trans_std:.2f} mm is high — result may be inaccurate")
    if t_norm < 200 or t_norm > 3000:
        warnings.append(f"|t|={t_norm:.0f} mm seems implausible")
    for w in warnings:
        print(f"  [WARN] {w}")
    if not warnings:
        print("  All checks passed.")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    np.savez(str(OUT_NPZ),
             T_cam2base=T_c2b, R_cam2base=R_c2b, t_cam2base=t_c2b,
             method=method, n_poses=N)
    print(f"\nSaved: {OUT_NPZ}")

    q = Rotation.from_matrix(R_c2b).as_quat()   # [x,y,z,w]
    rpy = Rotation.from_matrix(R_c2b).as_euler("xyz", degrees=True)
    out_dict = {
        "calibration_type": "eye_to_hand",
        "method":           method,
        "date":             str(datetime.date.today()),
        "n_poses":          int(N),
        "T_cam2base":       {"rows": 4, "cols": 4, "data": T_c2b.tolist()},
        "t_cam2base_mm":    (t_c2b.ravel() * 1000).tolist(),
        "quaternion_xyzw":  q.tolist(),
        "rpy_deg_xyz":      rpy.tolist(),
        "residuals": {
            "trans_std_mm": float(round(trans_std, 4)),
            "rot_std_deg":  float(round(rot_std,   4)),
        },
    }
    with open(OUT_YAML, "w") as f:
        yaml.dump(out_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Saved: {OUT_YAML}")
    print("\nNext step: python3 visual/obstacle_detector.py")


if __name__ == "__main__":
    main()
