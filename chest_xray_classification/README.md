# Optimization of Diagnostic Radiology Imaging Models for Edge TPUs

## Directory Structure

```bash
.
├── NIH_14
├── src
└── weights
```

- `NIH_14`: Download dataset from `https://www.kaggle.com/nih-chest-xrays/data` and extract in this directory.
- `src`: This contains source code for quantization aware training, serialization of data and inference on coral board.
- `weights`: This directory contains the edge tpu compiled weight file.

## Scripts Inside `src` Directory

```bash
src/
├── calculate_inference_time.py
├── calculate_metrics.ipynb
├── post_training_qunantization.py
├── quantization_aware_training.ipynb
├── quantize_models.ipynb
└── serialize_data.ipynb
```

- `calculate_inference_time.py`: This script calculate inference time on coral edge TPU. It also saves the prediction in a file which is used for metric calculation.

- `calculate_metrics.ipynb`: This notebook is used to calculate AUC score. This uses predictions from coral board and ground truth labels.

- `post_training_quantization.py:` This script contains utility to quantize the model to INT8.

- `quantization_aware_training.ipynb`: This notebook performs quantization aware training.

- `quantize_models.ipynb`: This notebook quantizes the models trained from quantization aware training to INT8.

- `serialize_data.ipynb`: This notebook saves the ground truth labels for test data and extracts test images for evaluation on coral board.