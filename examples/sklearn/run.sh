#!/usr/bin/env bash
# Train the model, then serve it as a REST API with mlup.
set -e

python train.py
mlup run -m model.pkl
