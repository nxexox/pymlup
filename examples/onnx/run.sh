#!/usr/bin/env bash
# Prepare a tiny ONNX model, then serve it as a REST API with mlup.
set -e

python prepare_model.py

# model.onnx has float32 inputs, but JSON numbers deserialize to Python
# int/float (float64) by default - dtype_for_predict=float32 makes mlup cast
# incoming request data to match what the ONNX model expects.
mlup run -m model.onnx --up.dtype_for_predict=float32
