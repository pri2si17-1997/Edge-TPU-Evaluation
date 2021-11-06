# Optimization of Ultrasound Nerve Segmentation Model for Edge TPU

## Directory Structure

```bash
.
├── dataset
├── src
└── weights
```

- `dataset`: Download `Ultrasound Nerve Segmentation` dataset in this directory. This directory contains a README with the link. 

- `src`: This directory contains the script for QAT (Quantization Aware Training) and inference on edge tpu.

- `weights`: This directory contains the INT8 and Edgetpu compiled weights.

## Scripts Inside `src` Directory

- `calculate_predictions.py`: This script computes the inference time and prediction on edge TPU. It will generate a file named `QAT_INT_8_Predictions.npy`. Use this file to create a submission using `QAT_INT8_Submission()` method in the `train_quantize_evaluate_nerve_segmentation.ipynb` notebook. This will create a csv file which can be uploaded to kaggle for score evaluation.

- `train_quantize_evaluate_nerve_segmentation.ipynb`: Use this notebook to perform Quantization Aware Training and INT 8 quantization of model. The baseline code is taken from <https://www.kaggle.com/gbatchkala/urss-2019-project-review>.

