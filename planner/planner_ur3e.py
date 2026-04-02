from __future__ import annotations

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

# ====== Internal imports =================================
CUR_PATH  = Path(__file__).resolve().parent          # .../planner/
ROOT_PATH = CUR_PATH.parent                          # .../three_freedom/
sys.path.insert(0, str(ROOT_PATH))                   # ur3e/, rdf/ etc. visible

from cdf.mlp     import MLPRegression
from cdf.nn_cdf  import CDF
from ur3e.ur3e   import URRobot
from rdf.ur_rdf  import BPSDF
from planner.obstacle_manager import CrossObstacleManager

# --- Reproducibility ---
SEED: int = 42

# --- Experiment scale / runtime ---
NUM_TRIALS: int = 10
MAX_STEPS: int = 3000
DT: float = 0.02
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE: bool = True

# --- Model checkpoint ---
MODEL_PATH: str | Path = ROOT_PATH / "cdf" / "model" / "model_best.pt"
MODEL_INDEX: int = 19900  # last checkpoint (only used for model_dict.pt)

# --- MPPI Sampling ---
ALG_N_SAMPLES: int = 200
ALG_HORIZON: int = 1
ALG_BETA: float = 1.0

# --- Control ---
CONS_U: float = 1.5  # control magnitude limit

# --- Cost weights ---
GOAL_ANGLE_COST_W: float = 10
OBS_ANGLE_PENALTY_W: float = 20
OBST_PENALTY_ANGLE_BASE: float = 90.0
OBST_PENALTY_DIST: float = 0.8

# --- Success condition ---
EPS_GOAL: float = 0.05

# --- Save directory ---
SAVE_DIR: str | Path = ROOT_PATH / "saved_traj_ur3e"

# Fixed start/goal pairs — only joint[0] differs by π rad; matches visualize_obstacles.py Q0/QF
FIXED_PAIRS = [
    (
        np.array([-1.57, -0.5,  1.2, -1.3, -1.57, 0.0], dtype=np.float32),
        np.array([ 1.57, -0.5,  1.2, -1.3, -1.57, 0.0], dtype=np.float32),
    ),
]

# --- UR3e joint limits (±2π for all 6 joints, matches ur3e.py theta_min/max) ---
JOINT_LIMITS = np.array(
    [
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
    ],
    dtype=np.float32,
)

# ==============================================================================
# Init reproducibility
# ==============================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PI = np.pi
np.set_printoptions(precision=4, suppress=True)
torch.set_default_dtype(torch.float32)


# ==============================================================================
# MPPI Planner for UR3e
# ==============================================================================

class MPPIUR3e:
    """MPPI planner for UR3e with CDF obstacle avoidance"""

    def __init__(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        obs_pts: torch.Tensor,
        cdf: CDF,
        net: nn.Module,
        rdf: BPSDF,
        rdf_model: nn.Module,
        dt: float = DT,
        cons_u: float = CONS_U,
        obs_radius: float = 0.05,
    ):
        self.start = start.detach().float()
        self.goal = goal.detach().float()
        self.obs_pts = obs_pts.float()
        self.cdf = cdf
        self.net = net
        self.rdf = rdf
        self.rdf_model = rdf_model
        self.dt = dt
        self.cons_u = cons_u
        self.obs_radius = obs_radius
        self.device = start.device

        self.q_lim = JOINT_LIMITS.copy()
        self.N_SAMPLES = ALG_N_SAMPLES
        self.HORIZON = ALG_HORIZON
        self.BETA = ALG_BETA

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Simple Euler integration"""
        return x + u * self.dt

    def cost_batch(self, x: torch.Tensor, u_batch: torch.Tensor) -> torch.Tensor:
        """Cost function: goal direction + obstacle penalty"""
        # Goal cost
        goal_vec = (self.goal - x)
        goal_norm = goal_vec.norm() + 1e-8
        move_norm = u_batch.norm(dim=-1) + 1e-8
        cos_dir = (u_batch @ goal_vec) / (move_norm * goal_norm)
        theta_deg = torch.acos(cos_dir.clamp(-1., 1.)) * 180. / PI
        cost = GOAL_ANGLE_COST_W * theta_deg

        # Obstacle penalty using CDF
        x_req = x.detach().clone().requires_grad_(True)
        dist, grad = self.cdf.inference_d_wrt_q(self.obs_pts, x_req.unsqueeze(0), self.net, True)
        dist_val = dist.item()
        grad_vec = grad.squeeze(0)
        grad_norm = grad_vec.norm().item()
        if (dist_val < OBST_PENALTY_DIST) and (grad_norm > 1e-6) and (dist_val < goal_norm.item()):
            grad_vec = grad_vec / (grad_norm + 1e-8)
            cos_obs = (u_batch @ grad_vec) / (move_norm * (grad_vec.norm() + 1e-8))
            theta_obs = torch.acos(cos_obs.clamp(-1., 1.)) * 180. / PI
            penalty = torch.clamp(theta_obs - OBST_PENALTY_ANGLE_BASE, min=0.)
            cost += OBS_ANGLE_PENALTY_W * penalty

        return cost

    def mppi_sample(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Sample control sequences"""
        flat_mean = mean.flatten()
        block_cov = torch.block_diag(*cov)
        dist = MultivariateNormal(flat_mean, block_cov)
        samples = dist.sample((self.N_SAMPLES,))
        u_all = samples.view(self.N_SAMPLES, self.HORIZON, 6).clamp_(-self.cons_u, self.cons_u)
        return u_all

    def update_mean_cov(
        self,
        mu_prev: torch.Tensor,
        Sigma_prev: torch.Tensor,
        x: torch.Tensor,
        u_all: torch.Tensor,
    ):
        """Update MPPI distribution"""
        w_cost = self.cost_batch(x, u_all[:, 0, :])
        w = torch.exp(-self.BETA * (w_cost - w_cost.min()))
        w_sum = w.sum()

        mu_new = (w[:, None, None] * u_all).sum(0) / w_sum
        diff = u_all - mu_new.unsqueeze(0)
        cov_w = torch.einsum('n,nhc,nhd->hcd', w, diff, diff) / w_sum

        mu_s = 0.5 * mu_prev + 0.5 * mu_new
        Sigma_s = 0.5 * Sigma_prev + 0.5 * cov_w + 1e-2 * torch.eye(6, device=x.device)
        return mu_new[0], mu_s, Sigma_s

    def optimise(
        self,
        max_steps: int = MAX_STEPS,
        eps_goal: float = EPS_GOAL,
        verbose: bool = False,
    ):
        """Run MPPI optimization"""
        device = self.start.device
        x_curr = self.start.squeeze(0)
        mean = torch.zeros(self.HORIZON, 6, device=device)
        cov = torch.eye(6, device=device).repeat(self.HORIZON, 1, 1)

        traj = [x_curr.detach().cpu().numpy()]
        total_path_len = 0.0
        steps_taken = 0

        t0 = time.perf_counter()

        for k in range(max_steps):
            # MPPI sampling & update
            u_all = self.mppi_sample(mean, cov)
            best_u, mean, cov = self.update_mean_cov(mean, cov, x_curr, u_all)

            # Execute control
            u_exec = best_u.clamp_(-self.cons_u, self.cons_u)
            x_next = self.dynamics(x_curr, u_exec)

            # Check joint limits
            x_next_np = x_next.detach().cpu().numpy()
            if np.any(x_next_np < self.q_lim[:, 0]) or np.any(x_next_np > self.q_lim[:, 1]):
                if verbose:
                    print(f"❌ step {k + 1}: joint limit violation")
                return traj, False, total_path_len, steps_taken, time.perf_counter() - t0

            # Collision check via RDF — subtract obstacle radius to get clearance
            pose = torch.eye(4).unsqueeze(0).to(self.device).float()
            sdf, _ = self.rdf.get_whole_body_sdf_batch(
                self.obs_pts, pose, x_next.unsqueeze(0),
                self.rdf_model, use_derivative=False
            )
            min_sdf = sdf.min().item() - self.obs_radius
            
            if min_sdf < 0:  # collision threshold
                if verbose:
                    print(f"❌ step {k + 1}: obstacle collision (sdf={min_sdf:.4f})")
                return traj, False, total_path_len, steps_taken, time.perf_counter() - t0

            total_path_len += torch.norm(x_next - x_curr).item()
            traj.append(x_next.detach().cpu().numpy())
            x_curr = x_next.detach()
            steps_taken = k + 1

            # Check goal
            if (x_curr - self.goal).norm().item() < eps_goal:
                return traj, True, total_path_len, steps_taken, time.perf_counter() - t0

        return traj, False, total_path_len, steps_taken, time.perf_counter() - t0


# ==============================================================================
# Experiment runner
# ==============================================================================

def run_experiments(
    num_trials: int = NUM_TRIALS,
    max_steps: int = MAX_STEPS,
    dt: float = DT,
    eps_goal: float = EPS_GOAL,
    cons_u: float = CONS_U,
    device: str = DEVICE,
    verbose: bool = VERBOSE,
    save_dir: str | Path = SAVE_DIR,
    model_path: str | Path = MODEL_PATH,
    model_index: int = MODEL_INDEX,
    fixed_pairs=None,   # list of (q0, qf) np arrays; None → random sampling
):
    device = torch.device(device)

    # Load CDF network
    cdf = CDF(device, load_data=False)
    net = MLPRegression(9, 1, [1024, 512, 256, 128, 128], skips=[], act_fn=nn.ReLU, nerf=True)

    state_obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_obj, dict) and all(isinstance(k, int) for k in state_obj.keys()):
        net.load_state_dict(state_obj[model_index])
    else:
        net.load_state_dict(state_obj)
    net.to(device).eval()

    print(f"Loaded CDF model from {model_path}, checkpoint {model_index}")
    
    # Load RDF model for collision detection
    robot = URRobot(device)
    rdf = BPSDF(8, -1.0, 1.0, robot, device)
    rdf_model = torch.load(
        ROOT_PATH / 'rdf/models/BP_8.pt',
        map_location=device,
        weights_only=False
    )
    print(f"Loaded RDF model for collision detection")

    # Setup obstacle
    obs_manager = CrossObstacleManager(sphere_radius=0.05, device=str(device))
    obs_pts = obs_manager.get_obstacle_points()
    obs_radius = obs_manager.sphere_radius
    print(f"Cross obstacle: {len(obs_pts)} spheres (r={obs_radius}m), "
          f"center≈[0.15,0.0,0.2], within workspace X[-0.4,0.4] Y[-0.4,0.4] Z[0,0.4]")

    # Save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    succ, steps_list, time_list, path_len_list = 0, [], [], []

    PLAN_THRESH = obs_radius + 0.02

    # Run trials
    for trial in range(1, num_trials + 1):
        pose = torch.eye(4).unsqueeze(0).to(device).float()

        if fixed_pairs is not None:
            pair = fixed_pairs[(trial - 1) % len(fixed_pairs)]
            q0, qf = pair[0].copy(), pair[1].copy()
            sdf0, _ = rdf.get_whole_body_sdf_batch(
                obs_pts, pose,
                torch.from_numpy(q0).float().to(device).unsqueeze(0),
                rdf_model, use_derivative=False)
            sdff, _ = rdf.get_whole_body_sdf_batch(
                obs_pts, pose,
                torch.from_numpy(qf).float().to(device).unsqueeze(0),
                rdf_model, use_derivative=False)
            if verbose:
                print(f"[trial {trial}] Fixed pair: "
                      f"start_sdf={sdf0.min().item():.3f}, goal_sdf={sdff.min().item():.3f}")
        else:
            # Random sampling — require that the straight-line path contains a collision
            max_attempts = 500
            for attempt in range(max_attempts):
                q0 = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]).astype(np.float32)
                qf = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]).astype(np.float32)

                # Start and goal must both be collision-free
                sdf0, _ = rdf.get_whole_body_sdf_batch(obs_pts, pose,
                    torch.from_numpy(q0).float().to(device).unsqueeze(0),
                    rdf_model, use_derivative=False)
                if sdf0.min().item() < PLAN_THRESH:
                    continue
                sdff, _ = rdf.get_whole_body_sdf_batch(obs_pts, pose,
                    torch.from_numpy(qf).float().to(device).unsqueeze(0),
                    rdf_model, use_derivative=False)
                if sdff.min().item() < PLAN_THRESH:
                    continue

                # At least one waypoint on the straight-line path must collide
                n_check = 20
                collision_detected = False
                for i in range(1, n_check - 1):
                    alpha = i / (n_check - 1)
                    q_interp = torch.from_numpy((1 - alpha) * q0 + alpha * qf).float().to(device)
                    sdf, _ = rdf.get_whole_body_sdf_batch(
                        obs_pts, pose, q_interp.unsqueeze(0),
                        rdf_model, use_derivative=False
                    )
                    if sdf.min().item() < PLAN_THRESH:
                        collision_detected = True
                        break

                if collision_detected:
                    if verbose:
                        print(f"[trial {trial}] Found valid pair (attempt {attempt + 1}): "
                              f"start_sdf={sdf0.min().item():.3f}, goal_sdf={sdff.min().item():.3f}")
                    break
            else:
                if verbose:
                    print(f"[trial {trial}] Warning: Could not find valid pair after {max_attempts} attempts")

        solver = MPPIUR3e(
            start=torch.tensor(q0).unsqueeze(0).to(device),
            goal=torch.tensor(qf).to(device),
            obs_pts=obs_pts,
            cdf=cdf,
            net=net,
            rdf=rdf,
            rdf_model=rdf_model,
            dt=dt,
            cons_u=cons_u,
            obs_radius=obs_radius,
        )

        traj, success, path_len, steps_taken, t_elapsed = solver.optimise(
            max_steps=max_steps,
            eps_goal=eps_goal,
            verbose=verbose,
        )

        traj_np = np.stack(traj, axis=0).astype(np.float32)

        # Analyze trajectory to check if it's avoiding obstacles
        if success:
            # Check minimum SDF along trajectory using RDF
            min_sdf_along_path = float('inf')
            for q in traj:
                q_t = torch.from_numpy(q).float().to(device)
                pose = torch.eye(4).unsqueeze(0).to(device).float()
                sdf, _ = rdf.get_whole_body_sdf_batch(
                    obs_pts, pose, q_t.unsqueeze(0),
                    rdf_model, use_derivative=False
                )
                min_sdf_along_path = min(min_sdf_along_path, sdf.min().item())
            
            # Compare with direct path
            direct_min_sdf = float('inf')
            for i in range(20):
                alpha = i / 19
                q_interp = torch.from_numpy((1 - alpha) * q0 + alpha * qf).float().to(device)
                pose = torch.eye(4).unsqueeze(0).to(device).float()
                sdf, _ = rdf.get_whole_body_sdf_batch(
                    obs_pts, pose, q_interp.unsqueeze(0),
                    rdf_model, use_derivative=False
                )
                direct_min_sdf = min(direct_min_sdf, sdf.min().item())

        out_file = save_dir / f"trial_{trial:04d}.npz"
        np.savez_compressed(out_file, q=traj_np, start=q0, goal=qf, success=success)
        
        if success:
            succ += 1
            steps_list.append(len(traj) - 1)
            time_list.append(t_elapsed)
            path_len_list.append(path_len)

        if verbose:
            status = '✓' if success else '✗'
            if success:
                avoidance_info = f"min_sdf_path={min_sdf_along_path:.4f} vs direct={direct_min_sdf:.4f}"
                print(
                    f"[trial {trial:3d}] {status} "
                    f"steps={len(traj) - 1:4d} time={t_elapsed:.3f}s "
                    f"path_len={path_len:.3f} {avoidance_info}"
                )
            else:
                print(
                    f"[trial {trial:3d}] {status} "
                    f"steps={len(traj) - 1:4d} time={t_elapsed:.3f}s "
                    f"path_len={path_len:.3f}"
                )

    # Statistics
    stats = {
        "success_rate": succ / num_trials if num_trials else 0.0,
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "std_steps": float(np.std(steps_list)) if steps_list else 0.0,
        "avg_time": float(np.mean(time_list)) if time_list else 0.0,
        "std_time": float(np.std(time_list)) if time_list else 0.0,
        "avg_path_len": float(np.mean(path_len_list)) if path_len_list else 0.0,
        "num_success": int(succ),
        "num_trials": int(num_trials),
    }

    # Save index
    index = {"success": [], "meta": {"dt": dt, "eps_goal": eps_goal}, "summary": stats}
    index_path = save_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print("\n========== Summary ==========")
    for k, v in stats.items():
        print(f"{k:>15}: {v:.3f}")

    print(f"\nSaved {succ} successful trajectories to: {save_dir}")
    print(f"Index written to: {index_path}")

    return stats


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    run_experiments(fixed_pairs=FIXED_PAIRS)
