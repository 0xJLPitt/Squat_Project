"""
Squat_with_AGFormer Tools Package
Contains modular utilities for 2D keypoint conversion, MotionAGFormer 3D pose inference,
2D/3D overlays, 3D animations, and angle/jitter diagnostic plotting.
"""

from .convert_keypoints import (
    interpolate_missing_keypoints,
    coco_to_h36m,
    turn_into_h36m,
    convert_txt_to_npz
)

from .infer_agformer import (
    load_agformer_model,
    infer_agformer_3d,
    infer_3d,
    turn_into_clips,
    resample,
    flip_data
)

from .overlay_2d import overlay_2d
from .overlay_yolo import overlay_yolo_pose
from .visualize_3d import visualize_npz, init_3d_pose, update_3d_pose

__all__ = [
    'interpolate_missing_keypoints',
    'coco_to_h36m',
    'turn_into_h36m',
    'convert_txt_to_npz',
    'load_agformer_model',
    'infer_agformer_3d',
    'infer_3d',
    'turn_into_clips',
    'resample',
    'flip_data',
    'overlay_2d',
    'overlay_yolo',
    'visualize_npz',
    'init_3d_pose',
    'update_3d_pose'
]
