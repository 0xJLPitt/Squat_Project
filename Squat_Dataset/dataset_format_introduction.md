```text
dataset/
├── splits/
│   ├── train.json
│   ├── val.json
│   └── test.json
│
├── taxonomy/
│   ├── error_definitions.json
│   └── camera_views.json
│
├── S001/
│   ├── subject_metadata.json
│   │
│   ├── session01/
│   │   ├── session_metadata.json
│   │   │
│   │   ├── calibration/
│   │   │   ├── extri.yml
│   │   │   └── intri.yml
│   │   │
│   │   ├── clips/
│   │   │   ├── CLIP_0001/
│   │   │   │   ├── video/
│   │   │   │   │   ├── L.mp4
│   │   │   │   │   ├── F.mp4
│   │   │   │   │   ├── FL.mp4
│   │   │   │   │   ├── RL.mp4
│   │   │   │   │   └── RRU.mp4
│   │   │   │   │
│   │   │   │   ├── coordinate/
│   │   │   │   │   ├── 2D/
│   │   │   │   │   │   ├── L.csv
│   │   │   │   │   │   ├── F.csv
│   │   │   │   │   │   └── ...
│   │   │   │   │   │
│   │   │   │   │   ├── 3D/
│   │   │   │   │   │   └── pose_3d.json
│   │   │   │   │   │
│   │   │   │   │   └── barbell/
│   │   │   │   │       └── barbell.csv
│   │   │   │   │
│   │   │   │   ├── angle/
│   │   │   │   │   ├── 2D/
│   │   │   │   │   └── 3D/
│   │   │   │   │
│   │   │   │   └── metadata.json
│   │   │   │
│   │   │   └── CLIP_0002/
│   │   │
│   │   └── notes.txt
│   │
│   └── session02/
│
└── S002/
```

### `subject_metadata.json`

```json
{
  "subject_id": "S001",
  "name": "Pitt",
  "gender": "male",
  "age_range": "30-35",
  "dom_hand": "right",
  "height": "177",
  "weight": "72",
  "test_item": "deadlift",
  "1rm": "130",
  "train_years": "10",
  "phone": "+886958865858",
  "email": "[pitthunag23@gmil.com]",
  "date": "2026-07-06"
}
```

### `session_metadata.json`

```json
{
  "session_id": "session01",
  "record_date": "2026-07-06",
  "camera_setup": "6cam",
  "fps": 30,
  "resolution": "1920x1080",
  "pose_estimator": "YoloV11",
  "3d_reconstruction_method": "triangulation",
  "location": "studio",
  "notes": "lighting slightly changed"
}
```

### `metadata.json`

```json
{
  "clip_id": "CLIP_0001",
  "subject_id": "S001",
  "session_id": "session01",
  "exercise": "deadlift",
  "rep_index": 1,
  "correct": false,
  "errors": [
    {
      "type": "lower_back_rounding",
      "severity": 4
    },
    {
      "type": "barbell_knee_collision",
      "severity": 2
    }
  ],
  "has_plate": true,
  "plate_weight_kg": 20,
  "views": ["L", "F", "FL", "RL", "RRU"],
  "modalities": ["video", "2d_pose", "3d_pose", "barbell", "joint_angle"],
  "num_frames": 124,
  "fps": 30,
  "start_frame": 1830,
  "end_frame": 1954,
  "annotation_version": "v1.0"
}
```

### `error_definitions.json`

```json
{
  "lower_back_rounding": {
    "display_name": "Lower Back Rounding",
    "exercise": ["deadlift"],
    "severity_range": [1, 4]
  },
  "wrist_collapse": {
    "display_name": "Wrist Collapse",
    "exercise": ["bench_press"],
    "severity_range": [1, 4]
  }
}
```

### `camera_views.json`

```json
{
  "L": "Left Lateral",
  "F": "Front",
  "FL": "Front Left",
  "RL": "Rear Left",
  "RRU": "Rear Right Upper"
}
```
