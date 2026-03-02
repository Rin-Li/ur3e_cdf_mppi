import torch
import trimesh
import os
import numpy as np

# Full joint limits (±360°)
theta_min = [-6.28318530718] * 6
theta_max = [ 6.28318530718] * 6

link_order = ['base', 'shoulder', 'upperarm', 'forearm', 'wrist1', 'wrist2', 'wrist3']

visual_offset = {
    'base':     (0, 0, 3.14159265359, 0, 0, 0),
    'shoulder': (0, 0, 3.14159265359, 0, 0, 0),
    'upperarm': (1.57079632679, 0, -1.57079632679, 0, 0, 0.12),
    'forearm':  (1.57079632679, 0, -1.57079632679, 0, 0, 0.027),
    'wrist1':   (1.57079632679, 0, 0, 0, 0, -0.104),
    'wrist2':   (0, 0, 0, 0, 0, -0.08535),
    'wrist3':   (1.57079632679, 0, 0, 0, 0, -0.0921),
}

kinematic = {
    'base'      : (0, 0, 3.14159265359, 0, 0, 0),
    'shoulder'  : (0, 0, 0, 0, 0, 0.15185),
    'upperarm'  : (1.57079632679, 0, 0, 0, 0, 0),
    'forearm'   : (0, 0, 0, -0.24355, 0, 0),
    'wrist1'    : (0, 0, 0, -0.2132, 0, 0.13105),
    'wrist2'    : (1.57079632679, 0, 0, 0, -0.08535, 0),
    'wrist3'    : (1.57079632679, 3.14159265359, 3.14159265359, 0, 0.0921, 0),
}


class URRobot(torch.nn.Module):
    """
    UR3e forward kinematics and mesh forward pass.
    get_transformations_each_link is fully vectorized (no Python for-loop over the batch dimension).
    """
    def __init__(self, device='cpu', mesh_path=None):
        super().__init__()
        if mesh_path is None:
            mesh_path = os.path.join(os.path.dirname(__file__), 'model')
        self.device    = device
        self.mesh_path = mesh_path

        self.theta_max = torch.tensor(theta_max, device=device, dtype=torch.float32)
        self.theta_min = torch.tensor(theta_min, device=device, dtype=torch.float32)

        # Precompute constant 4x4 FK matrices from kinematic and visual_offset tables
        self._precompute_fk_constants(device=device, dtype=torch.float32)

        # load meshes
        self.meshes = self.load_mesh()
        self.robot, self.robot_faces, self.robot_normals = zip(*[
            self.meshes[link] for link in link_order
        ])

    # -------------------------
    # Basic transforms (4x4)
    # -------------------------
    def Rx(self, r):
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32, device=self.device)
        else:
            r = r.to(self.device)

        c, s = torch.cos(r), torch.sin(r)
        zero = torch.zeros_like(c)
        one = torch.ones_like(c)

        return torch.stack([
            torch.stack([one,  zero, zero, zero]),
            torch.stack([zero, c,   -s,   zero]),
            torch.stack([zero, s,    c,   zero]),
            torch.stack([zero, zero, zero, one ])
        ])

    def Ry(self, p):
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32, device=self.device)
        else:
            p = p.to(self.device)

        c, s = torch.cos(p), torch.sin(p)
        zero = torch.zeros_like(c)
        one = torch.ones_like(c)

        return torch.stack([
            torch.stack([c,    zero, s,    zero]),
            torch.stack([zero, one,  zero, zero]),
            torch.stack([-s,   zero, c,    zero]),
            torch.stack([zero, zero, zero, one ])
        ])

    def Rz(self, y):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(self.device)

        c, s = torch.cos(y), torch.sin(y)
        zero = torch.zeros_like(c)
        one = torch.ones_like(c)

        return torch.stack([
            torch.stack([c,   -s,   zero, zero]),
            torch.stack([s,    c,   zero, zero]),
            torch.stack([zero, zero, one,  zero]),
            torch.stack([zero, zero, zero, one ])
        ])

    def T(self, x, y, z):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)

        x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)

        return torch.stack([
            torch.stack([one,  zero, zero, x]),
            torch.stack([zero, one,  zero, y]),
            torch.stack([zero, zero, one,  z]),
            torch.stack([zero, zero, zero, one])
        ])

    def T_origin(self, roll, pitch, yaw, x, y, z):
        # T(x,y,z) @ Rz(yaw) @ Ry(pitch) @ Rx(roll)
        return (self.T(x, y, z) @ self.Rz(yaw) @ self.Ry(pitch) @ self.Rx(roll))

    # -------------------------
    # Precompute FK constants
    # -------------------------
    def _precompute_fk_constants(self, device="cpu", dtype=torch.float32):
        T_kin = [self.T_origin(*kinematic[link]) for link in link_order]         # 7*(4,4)
        T_off = [self.T_origin(*visual_offset[link]) for link in link_order]     # 7*(4,4)

        T_kin = torch.stack(T_kin, dim=0).to(device=device, dtype=dtype)         # (7,4,4)
        T_off = torch.stack(T_off, dim=0).to(device=device, dtype=dtype)         # (7,4,4)

        # Register as buffers so they move with model.to(device)
        self.register_buffer("T_kin", T_kin, persistent=False)
        self.register_buffer("T_off", T_off, persistent=False)

    # -------------------------
    # Batch Rz for joints
    # -------------------------
    def Rz_batch(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: (B,6)
        return: (B,6,4,4)
        No in-place ops; autograd-safe.
        """
        assert theta.ndim == 2 and theta.shape[1] == 6
        c = torch.cos(theta)
        s = torch.sin(theta)
        zeros = torch.zeros_like(c)
        ones  = torch.ones_like(c)

        row0 = torch.stack([c,   -s,  zeros, zeros], dim=-1)
        row1 = torch.stack([s,    c,  zeros, zeros], dim=-1)
        row2 = torch.stack([zeros, zeros, ones,  zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, zeros, ones ], dim=-1)

        R = torch.stack([row0, row1, row2, row3], dim=-2)  # (B,6,4,4)
        return R


    def get_transformations_each_link(self, pose: torch.Tensor, theta: torch.Tensor):
        """
        pose:  (B,4,4)
        theta: (B,6)
        return: list of 7 tensors, each (B,4,4)
        No in-place ops; autograd-safe.
        """
        assert pose.ndim == 3 and pose.shape[-2:] == (4, 4)
        assert theta.ndim == 2 and theta.shape[1] == 6
        assert pose.shape[0] == theta.shape[0]

        device, dtype = theta.device, theta.dtype

        T_kin = self.T_kin.to(device=device, dtype=dtype)  # (7,4,4)
        T_off = self.T_off.to(device=device, dtype=dtype)  # (7,4,4)

        Rz = self.Rz_batch(theta)  # (B,6,4,4)

        # base transform
        cur = pose @ T_kin[0]              # (B,4,4)

        out = [cur @ T_off[0]]             # base link output

        # joints (1..6)
        for idx in range(1, 7):
            rel = T_kin[idx].unsqueeze(0) @ Rz[:, idx - 1]   # (B,4,4)
            cur = cur @ rel
            out.append(cur @ T_off[idx])

        return out


    # -------------------------
    # Mesh forward utilities
    # -------------------------
    def _transform_vn(self, v, n, T, B):
        # v: (Nv,4), n: (Nv,4), T: (B,4,4)
        v = v.repeat(B, 1, 1)
        n = n.repeat(B, 1, 1)
        v = (T @ v.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        n = (T @ n.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        return v, n

    def forward(self, pose, theta):
        B = theta.size(0)
        T_link = self.get_transformations_each_link(pose, theta)
        verts, norms = [], []
        for i, _ in enumerate(link_order):
            v, n = self._transform_vn(self.robot[i], self.robot_normals[i], T_link[i], B)
            verts.append(v)
            norms.append(n)
        return verts + norms

    def _make_mesh_batch(self, v_list, f_list):
        return [
            trimesh.Trimesh(vertices=v.cpu().numpy(), faces=f.cpu().numpy(), process=False)
            for v, f in zip(v_list, f_list)
        ]

    def get_forward_robot_mesh(self, pose, theta):
        B = pose.size(0)
        out = self.forward(pose, theta)
        verts = out[:7]
        batches = []
        for b in range(B):
            meshes = self._make_mesh_batch([v[b] for v in verts], self.robot_faces)
            batches.append(trimesh.util.concatenate(meshes))
        return batches

    def load_mesh(self):
        meshes = {}
        for f in os.listdir(self.mesh_path):
            if not f.endswith('.stl'):
                continue
            m = trimesh.load(os.path.join(self.mesh_path, f))
            name = os.path.splitext(f)[0]

            ones = torch.ones(len(m.vertices), 1)
            v = torch.tensor(m.vertices, dtype=torch.float32)
            n = torch.tensor(m.vertex_normals, dtype=torch.float32)

            meshes[name] = [
                torch.cat((v, ones), -1).to(self.device),                      # verts_h: (Nv,4)
                torch.tensor(m.faces, dtype=torch.long).to(self.device),        # faces: (Nf,3)
                torch.cat((n, ones), -1).to(self.device)                       # normals_h: (Nv,4)
            ]
        return meshes


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ur = URRobot(device=device)

    # example
    theta = torch.zeros(1, 6, device=device, dtype=torch.float32)
    pose  = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        mesh = ur.get_forward_robot_mesh(pose, theta)[0]

    os.makedirs('output_meshes', exist_ok=True)
    mesh.export('output_meshes/ur3e_zero_pose.stl')
    print('Export to output_meshes/ur3e_zero_pose.stl')


if __name__ == '__main__':
    main()
