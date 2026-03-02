import torch


class CrossObstacleManager:
    """Cross-shaped obstacle composed of sphere arrays, placed within the training workspace X[-0.4,0.4] Y[-0.4,0.4] Z[0,0.4]."""

    def __init__(self, sphere_radius: float = 0.05, device: str = "cuda"):
        self.sphere_radius = sphere_radius
        self.device = torch.device(device)

        centers = self.pts(
            center_x=0.3, center_y=0.0, center_z=0.2,
            n_y=2, n_z=2, step_y=0.025, step_z=0.025
        )
        self.centers0 = centers.clone()
        self.centers  = centers.clone()

    def pts(self, center_x=0.15, center_y=0.0, center_z=0.2,
            n_y=2, n_z=2, step_y=0.025, step_z=None) -> torch.Tensor:
        """Generate sphere centers for the cross shape: horizontal arm along Y, vertical arm along Z."""
        device = self.device
        dtype  = torch.float32
        if step_z is None:
            step_z = step_y

        cx = torch.tensor(center_x, device=device, dtype=dtype)
        cy = torch.tensor(center_y, device=device, dtype=dtype)
        cz = torch.tensor(center_z, device=device, dtype=dtype)

        offs_y = (torch.arange(n_y, device=device, dtype=dtype) - (n_y - 1) / 2.0) * step_y
        offs_z = (torch.arange(n_z, device=device, dtype=dtype) - (n_z - 1) / 2.0) * step_z

        # Horizontal arm: spheres spaced along Y
        centers_y = torch.stack([
            cx.expand(n_y), cy + offs_y,
            cz.expand(n_y)
        ], dim=-1)
        # Vertical arm: spheres spaced along Z
        centers_z = torch.stack([
            cx.expand(n_z), cy.expand(n_z),
            cz + offs_z
        ], dim=-1)

        return torch.cat([centers_y, centers_z], dim=0)

    def get_obstacle_points(self) -> torch.Tensor:
        """Return sphere centers as the obstacle point cloud in task space."""
        return self.centers.float().to(self.device)

    @property
    def obstacle_radius(self):
        return self.sphere_radius
