# Model Card: PatchTST for Sports Motion Analysis

This model card provides detailed information about the PatchTST-based classification model used for detecting postural errors in exercises like Deadlift and Benchpress.

## Model Details

- **Model Name:** PatchTST Classifier for Sports Form Analysis
- **Model Type:** Time Series Transformer (PatchTST)
- **Developer:** CatsLab (National Cheng Kung University)
- **Date:** March 2026
- **Task:** Multi-label classification of exercise technique and error detection.
- **Supported Sports:** Deadlift, Benchpress.

## Architecture

The model is based on the **PatchTST (Patch Time Series Transformer)** architecture, which is specifically designed to handle longitudinal or sequential data by segmenting the time series into "patches" and processing them as tokens in a Transformer.

- **Patch Length:** 10
- **Embedding Dimension:** 256
- **Number of Attention Heads:** 4
- **Number of Encoder Layers:** 4
- **Dropout Rate:** 0.3
- **Classification Head:** LayerNorm followed by a Linear layer mapping to the number of classes.

## Intended Use

- **Primary Use:** Automated analysis of resistance training form using pose estimation data (2D/3D coordinates or joint angles).
- **Primary Users:** Sports biomechanics researchers, fitness coaching app developers, and strength & conditioning professionals.
- **Out-of-Scope Use:** Medical diagnosis of injuries or real-time clinical gait analysis without proper validation.

## Training Data

The model is trained on specialized sports datasets:
1. **MOCVD (Deadlift):** 1,109 sets including 8,506 repetitions from 156 subjects. Includes multi-view camera data and 3D reconstructed keypoints.
2. **Benchpress Dataset:** JSON-formatted trajectory data focusing on wrist and barbell movements.

### Augmentation Techniques
To improve robustness, the following augmentations are applied during training:
- **Time Stretching:** Randomly stretching or compressing the sequence length by a factor of 0.8x to 1.2x.
- **Gaussian Noise:** Adding subtle noise ($\sigma=0.01$) to the input coordinates to simulate sensor or estimation variance.

## Training Procedure

- **Loss Function:** `BCEWithLogitsLoss` (supports multi-label detection for overlapping errors).
- **Optimizer:** Adam ($LR = 10^{-4}$).
- **Learning Rate Scheduler:** Warmup Cosine Decay (5 warmup epochs, 100 total epochs).
- **Input Length:** 110 frames (Deadlift), 100 frames (Benchpress).
- **Hardware:** Trained on NVIDIA GPUs via CUDA.

## Metrics & Evaluation

The model's performance is evaluated using:
- **Macro F1-score:** To account for class imbalance (crucial for error detection where 'Correct' samples often dominate).
- **Accuracy:** Overall prediction accuracy across all labels.
- **Inference Time:** Measured in seconds per sample.
- **Multi-label Confusion Matrix:** Providing class-specific False Positive and False Negative rates.

## Factors & Bias

- **Subject Variance:** Performance may vary based on subject height, weight, and individual movement styles.
- **Camera View:** The MOCVD dataset is multi-view; however, performance might decrease if only an unoptimized or obstructed view is used.
- **Pose Estimation Quality:** The model is a "downstream" classifier; its performance is highly dependent on the accuracy of the upstream pose estimator (e.g., YOLOv11-Pose).

## Ethical Considerations

- **Privacy:** Training data involves human subjects. All data used should be anonymized and handled in accordance with institutional privacy standards.
- **Safety:** AI-based feedback should not replace professional supervision for beginners performing high-load resistance training.

## Caveats and Recommendations

- **Normalization:** Ensure that input sequences are normalized (Min-Max or Z-score) as expected by the model's preprocessing pipeline.
- **Thresholding:** The default classification threshold is 0.5. For critical error detection (where safety is paramount), practitioners may consider lowering the threshold to favor Recall over Precision.
