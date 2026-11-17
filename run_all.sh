#!/usr/bin/env bash
# Runs the whole pipeline in order.
set -e
python generate_dataset.py
python preprocess.py
python train_mlp.py
python evaluate_model.py
python convert_tflite.py
