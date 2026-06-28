# DLmodel_train

This repository contains a PyTorch-based deep learning training and evaluation pipeline designed for sports motion analysis and error detection. It primarily focuses on evaluating form and detecting errors in exercises like **Deadlift** and **Benchpress**.

The project uses advanced time-series and sequential deep learning models to classify workout techniques as correct or identify specific types of postural errors based on human pose/coordinate data.

## Features

- **Multi-Sport Support**: Configured to train and test on different exercises via command-line arguments (`--sport deadlift`, `--sport benchpress`).
- **Diverse Model Architectures**: Includes several types of neural network architectures tailored for temporal/sequential data:
  - **PatchTST**: A Transformer-based model optimized for time-series forecasting and classification.
  - **ResNet32**: 1D Convolutional Residual Network for extracting temporal features.
  - **LSTM & BiLSTM**: Recurrent Neural Networks for baseline sequence modeling.
- **Comprehensive Evaluation**: Evaluates models using Macro F1-score, Accuracy, and generates customizable Multi-Label Confusion Matrices.
- **Cross-Validation**: Supports robust dataset splitting including 6-fold cross-validation (`--split_training`).
- **Custom LR Scheduling**: Implements Warmup Cosine Annealing Learning Rate Scheduler for optimized training.

## Project Structure

```text
DLmodel_train/
├── PatchTST_train.py      # Main training script for PatchTST models
├── PatchTST_test.py       # Evaluation and testing script
├── main.py                # Additional runner/utility script
├── models.py              # Neural network definitions (PatchTST, ResNet32, LSTM)
├── tools.py               # Utilities for metric calculation, visualizations, and logging
├── dataset/               # PyTorch Dataset/DataLoader implementations
├── data/                  # Directory containing 'deadlift' and 'benchpress' datasets
├── models/                # Directory for saving trained model weights (.pth)
└── environment.yaml       # Conda environment dependency file
```

## Setup & Installation

The project uses `conda` for environment management. You can easily recreate the required environment using the provided `environment.yaml` file.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda if you haven't already.
2. Create the environment:
    ```bash
    conda env create -f environment.yaml
    ```
3. Activate the environment:
    ```bash
    conda activate deadlift_env
    ```

## Usage

### Training

To train the PatchTST model, run `PatchTST_train.py`. You can specify the sport and whether to use split training (k-fold).

```bash
# Train for deadlift
python PatchTST_train.py --sport deadlift

# Train for benchpress with k-fold cross-validation
python PatchTST_train.py --sport benchpress --split_training True
```

Training will log progress to the console, save the best model weights (based on validation F1-score) to the `models/` directory, and generate a plot of Training/Validation metrics across epochs.

### Testing / Evaluation

To evaluate a trained model and generate Multi-label Confusion Matrices, run `PatchTST_test.py`.

```bash
# Test the deadlift model
python PatchTST_test.py --sport deadlift

# Test the benchpress model
python PatchTST_test.py --sport benchpress
```

The script will test the data over predefined random seeds, print the macro F1-score, accuracy, inference cost time, and save the generated confusion matrix plots to the respective results directories.

## Datasets

The models heavily rely on coordinate and temporal feature data. The datasets should be placed in the `data/` directory, organized into `deadlift` and `benchpress` subdirectories.

### Deadlift Dataset (MOCVD)
Utilizes the **Multiple Outlook-Angles Conventional Deadlift Dataset (MOCVD)**. Please check `DeadliftDataset/README_MOCVD.md` for specific details regarding the structure, modalities, and features.

**Download Links (Split 7z Archives):**
- [DeadliftDataset.7z.001](https://catslab.ee.ncku.edu.tw/public/DeadliftDataset/DeadliftDataset.7z.001)
- [DeadliftDataset.7z.002](https://catslab.ee.ncku.edu.tw/public/DeadliftDataset/DeadliftDataset.7z.002)
- [DeadliftDataset.7z.003](https://catslab.ee.ncku.edu.tw/public/DeadliftDataset/DeadliftDataset.7z.003)
- [DeadliftDataset.7z.004](https://catslab.ee.ncku.edu.tw/public/DeadliftDataset/DeadliftDataset.7z.004)
- [DeadliftDataset.7z.005](https://catslab.ee.ncku.edu.tw/public/DeadliftDataset/DeadliftDataset.7z.005)
- [DeadliftDataset.7z.006](https://catslab.ee.ncku.edu.tw/public/DeadliftDataset/DeadliftDataset.7z.006)

*Note: You will need to download all parts and use a tool like 7-Zip to extract them.*

### Benchpress Dataset
Utilizes JSON-formatted pre-extracted feature data (tracking wrist/barbell trajectories or body postures).

**Download Link:**
- [BenchpressDataset.zip](https://catslab.ee.ncku.edu.tw/public/BenchpressDataset.zip)

## Data Structure
Before running the training or testing scripts, ensure your `data/` directory is organized as follows:
```text
data/
├── deadlift/
│   └── (Extracted MOCVD dataset files)
└── benchpress/
    └── (Extracted Benchpress dataset files)
```

## License

Please refer to dataset-specific directives (e.g., MOCVD dataset is for academic research purposes only) and project authors regarding the code usage license.
