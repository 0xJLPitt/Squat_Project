"""
compare_wrist_bar package
Tools for comparing AGFormer 3D wrist midpoint trajectories vs YOLO detected barbell trajectories.
"""

from .compare_wrist_vs_bar import run_comparison
from .compare_calibrated_wrist_vs_bar import run_calibrated_comparison

__all__ = [
    'run_comparison',
    'run_calibrated_comparison'
]
