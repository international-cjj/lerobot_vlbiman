from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class OrientationEstimate:
    angle_rad: float | None
    angle_deg: float | None
    centroid_px: list[float] | None
    major_axis_px: float | None
    minor_axis_px: float | None
    covariance: list[list[float]] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrientationMomentsEstimator:
    def estimate(self, mask: np.ndarray) -> OrientationEstimate:
        binary_mask = (mask > 0).astype(np.float64)
        m00 = float(binary_mask.sum())
        if m00 <= 0.0:
            return OrientationEstimate(None, None, None, None, None, None)

        ys, xs = np.indices(binary_mask.shape, dtype=np.float64)
        x_bar = float((xs * binary_mask).sum() / m00)
        y_bar = float((ys * binary_mask).sum() / m00)
        x_centered = xs - x_bar
        y_centered = ys - y_bar
        mu20 = float((binary_mask * x_centered * x_centered).sum() / m00)
        mu02 = float((binary_mask * y_centered * y_centered).sum() / m00)
        mu11 = float((binary_mask * x_centered * y_centered).sum() / m00)
        covariance = np.asarray([[mu20, mu11], [mu11, mu02]], dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        principal = eigenvectors[:, 0]
        angle_rad = float(np.arctan2(principal[1], principal[0]) % (2.0 * np.pi))
        return OrientationEstimate(
            angle_rad=angle_rad,
            angle_deg=float(np.degrees(angle_rad) % 360.0),
            centroid_px=[x_bar, y_bar],
            major_axis_px=float(np.sqrt(max(eigenvalues[0], 0.0)) * 2.0),
            minor_axis_px=float(np.sqrt(max(eigenvalues[1], 0.0)) * 2.0),
            covariance=covariance.tolist(),
        )
