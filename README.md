# S-Parameter Cascade Correction Model

Machine learning models for correcting S-parameter cascade simulation errors in RF/microwave filter design.

## Overview

This project addresses the accumulation of errors when cascading S-parameters of individual RF filter subdevices (simulated individually in HFSS). Two complementary approaches are used:

1. **Subdevice Surrogate Models**: MLP-based forward models that replace HFSS for individual subdevice S-parameter prediction (single_step, double_step, open_stub, t_stub, tapered_line).

2. **Cascade Correction Model**: A Transformer-based model that learns to correct the cascaded S-parameters by taking the individual subdevice S-parameters as input and predicting the error-compensated full cascade response.

## Measured Data

The `dataset/measured/` directory contains measured S-parameter data along with the corresponding cascade sequences and combinations, optimized by the Genetic Algorithm (GA).

> **Note**: The GA optimization method is not the invention of this project's author. Therefore, this repository only provides the model-related implementation. Readers interested in the GA method can implement it themselves based on the referenced paper, or contact the author for details.

Each `.json` file defines the cascade sequence (`sequence`) and the geometric parameters of each subdevice (`devices`), while the corresponding `.csv` file contains the full measured S-parameter data.

## Project Structure

```
.
├── dataset/                 # Training/validation data
│   ├── cascade/             # Cascade samples (train.json, test.json)
│   ├── single_step/         # Single-step subdevice data
│   ├── double_step/         # Double-step subdevice data
│   ├── open_stub/           # Open stub subdevice data
│   ├── t_stub/              # T-stub subdevice data
│   └── tapered_line/        # Tapered line subdevice data
├── checkpoints/             # Trained models
│   ├── single_step/
│   ├── double_step/
│   ├── open_stub/
│   ├── t_stub/
│   ├── tapered_line/
│   └── cascaded/            # Cascade correction model
├── model_block.py           # Subdevice surrogate model training
├── model.py                 # Cascade correction model architecture
├── dataset.py               # Dataset loading and preprocessing
├── train.py                 # Cascade correction model training
├── predict.py               # Prediction with PDF output
├── test.py                  # Evaluation and error statistics
├── loss.py                  # Custom loss functions
├── utils.py                 # Utility functions
├── precompute_predicted_sparams.py  # Precompute surrogate model predictions
├── requirements.txt         # Dependencies
└── .gitignore
```

## Data Format

### Subdevice Data (`dataset/<family>/train.json`)
Each JSON array entry contains:
- `n`: unique device ID
- `x_order`: column names for the input
- `x`: input array (N_freq, n_features) = [f_ghz, geometry_params...]
- `y_order`: column names for the output
- `y`: output array (N_freq, 8) = [s11_re, s11_im, s21_re, s21_im, s22_re, s22_im, s12_re, s12_im]

### Cascade Data (`dataset/cascade/train.json`)
Each JSON array entry contains:
- `n`: unique sample ID
- `sequence`: list of (family, id) pairs defining the cascade order
- `s`: cascade S-parameter array (N_freq, 9) = [freq, s11_re, s11_im, s21_re, s21_im, s22_re, s22_im, s12_re, s12_im]

## Usage

### Train Subdevice Surrogate Models
```bash
python model_block.py --family all
```

### Train Cascade Correction Model
```bash
python train.py --data_dir ./dataset
```

### Evaluate Model
```bash
python test.py --model_dir ./checkpoints/cascaded --data_dir ./dataset
```

### Run Prediction
```bash
# Single sample
python predict.py --n 1 --model_dir ./checkpoints/cascaded --data_dir ./dataset

# Batch prediction (samples 10-20)
python predict.py --n 10-20 --model_dir ./checkpoints/cascaded --data_dir ./dataset --output_dir ./figures
```

### Precompute Predicted S-Parameters
```bash
# Run surrogate models on all unique devices, cache results
python precompute_predicted_sparams.py
```

## Models

### Subdevice Surrogate (`model_block.py`)
- Fourier feature encoding
- Residual MLP
- Physical soft constraints: passivity penalty, reciprocity penalty
- Cosine annealing LR, early stopping

### Cascade Correction (`model.py`)
- Residual Conv1d device encoder (9ch input: 8ch RI + 1ch port width diff)
- Frequency-axis self-attention (multi-layer Pre-LN Transformer)
- Device-axis cross-attention (bidirectional device ↔ freq)
- Learnable sine frequency encoding
- Multi-level fusion Conv1d → 4-channel output (S11, S21, S22, S12 dB)

## Citation

If you use this code in your research, please cite the project.
