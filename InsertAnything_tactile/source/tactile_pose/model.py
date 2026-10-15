"""Shared CNN -> ordered stereo concatenation + actual opening -> MLP -> SO(3)."""
import torch
from torch import nn
from .rotations import matrix_to_rotation_6d, rotation_6d_to_matrix


class TactileRotationNet(nn.Module):
    def __init__(self, feature_dim=128, opening_scale_m=0.08):
        super().__init__()
        layers = []
        for index, (a, b) in enumerate(zip((3, 24, 48, 96), (24, 48, 96, 128))):
            kernel = 5 if index == 0 else 3
            layers.extend((nn.Conv2d(a, b, kernel, stride=2, padding=kernel // 2),
                           nn.GroupNorm(8, b), nn.ReLU(inplace=True)))
        layers.extend((nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
                       nn.Linear(128 * 3 * 4, feature_dim), nn.ReLU(inplace=True)))
        self.encoder = nn.Sequential(*layers)
        self.fusion = nn.Sequential(nn.Linear(2 * feature_dim + 1, 256), nn.ReLU(inplace=True),
                                    nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 6))
        self.register_buffer("opening_scale_m", torch.tensor(float(opening_scale_m)))
        nn.init.normal_(self.fusion[-1].weight, std=1e-4)
        self.initialize_rotation(torch.eye(3))

    def initialize_rotation(self, rotation):
        """Use the training-only mean rotation as an initial regression bias."""
        with torch.no_grad():
            self.fusion[-1].bias.copy_(matrix_to_rotation_6d(rotation))

    def forward_6d(self, left_rgb, right_rgb, opening_m):
        """RGB: float BCHW in [0,1]; opening: actual joint sum, shape [B,1], m."""
        if left_rgb.shape != right_rgb.shape or left_rgb.ndim != 4 or left_rgb.shape[1] != 3:
            raise ValueError("Expected two equally shaped BCHW RGB tensors")
        batch = left_rgb.shape[0]
        encoded = self.encoder(torch.cat((left_rgb, right_rgb), dim=0) * 2 - 1)
        left, right = encoded[:batch], encoded[batch:]
        opening = opening_m.reshape(batch, 1) / self.opening_scale_m * 2 - 1
        return self.fusion(torch.cat((left, right, opening), dim=-1))

    def forward(self, left_rgb, right_rgb, opening_m):
        return rotation_6d_to_matrix(self.forward_6d(left_rgb, right_rgb, opening_m))
