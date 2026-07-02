import numpy as np


def generate_triangle_coverage(area_size: float = 100,
                                coverage_radius: float = 35,
                                spacing: float = 1.2) -> list:
    """Return three BS positions in a triangle layout centred in the area."""
    spacing = 1.2 * coverage_radius
    center_x, center_y = area_size / 2, area_size / 2
    return [
        (center_x - spacing / 2, center_y - spacing / (2 * np.sqrt(3))),   # bottom-left
        (center_x + spacing / 2, center_y - spacing / (2 * np.sqrt(3))),   # bottom-right
        (center_x,               center_y + spacing / np.sqrt(3)),          # top
    ]