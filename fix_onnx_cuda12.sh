#!/bin/bash
# Script to fix and install CUDA 12 compatible onnxruntime-gpu on Linux OS (Modal)

echo "[1/3] Uninstalling existing onnxruntime packages..."
pip uninstall -y onnxruntime onnxruntime-gpu

echo "[2/3] Installing NVIDIA CUDA 12 packages from pip..."
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12

echo "[3/3] Installing latest CUDA 12 compatible onnxruntime-gpu..."
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple

echo "============================================="
echo "Done! Please run your application with:"
echo "python app.py"
echo "============================================="
