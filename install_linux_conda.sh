#!/bin/bash
# Install script for Linux (Debian/Ubuntu) using Conda/Miniconda

# Exit on error
set -e

echo "=========================================================="
echo " Starting httfacelond Linux (Conda) Installation"
echo "=========================================================="

echo "[1/6] Installing Linux system dependencies (FFmpeg, GCC compiler, OpenCV libraries)..."
if command -v sudo &> /dev/null && [ -w /etc/apt/sources.list ]; then
    echo "[Info] Sudo available. Installing system packages via apt..."
    sudo apt-get update || true
    sudo apt-get install -y ffmpeg build-essential libgl1-mesa-glx libglib2.0-0 wget || true
else
    echo "[Warning] Sudo privileges not found or cannot write. Skip system packages, will install locally via Conda..."
fi

# Check if conda command exists
if ! command -v conda &> /dev/null; then
    echo "=========================================================="
    echo " Conda not found! Installing Miniconda automatically..."
    echo "=========================================================="
    
    # Download installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_installer.sh
    
    # Run installer silently
    bash miniconda_installer.sh -b -p $HOME/miniconda3
    
    # Clean up
    rm miniconda_installer.sh
    
    # Load conda shell hook for this script
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    echo "[Info] Miniconda installed successfully!"
else
    # Initialize conda shell support for existing installation
    eval "$(conda shell.bash hook)"
fi

# Accept Anaconda Terms of Service automatically
echo "[Info] Accepting Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "=========================================================="
echo "[2/6] Creating 'httfacelond' Conda environment (Python 3.12)..."
echo "=========================================================="
conda create -y -n httfacelond python=3.12
conda activate httfacelond

# Install tools inside conda environment if apt-get was skipped
if ! command -v sudo &> /dev/null || ! [ -w /etc/apt/sources.list ]; then
    echo "[Info] Installing FFmpeg inside Conda env..."
    conda install -y -c conda-forge ffmpeg
fi

echo "=========================================================="
echo "[3/6] Installing Python dependencies..."
echo "=========================================================="
pip install --upgrade pip
pip install numpy==1.26.4
pip install -r requirements.txt

echo "=========================================================="
echo "[4/6] Installing ONNX Runtime GPU..."
echo "=========================================================="
pip uninstall -y onnxruntime
pip install onnxruntime-gpu

echo "=========================================================="
echo "[5/6] Installing NVIDIA CUDA 12 libraries for GPU acceleration..."
echo "=========================================================="
pip install nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12

# Set LD_LIBRARY_PATH permanently in conda environment
NVIDIA_LIB_PATH=$(find $CONDA_PREFIX/lib/python3.12/site-packages/nvidia -name "lib" -type d 2>/dev/null | paste -sd:)
conda env config vars set LD_LIBRARY_PATH=${NVIDIA_LIB_PATH}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
echo "[Info] LD_LIBRARY_PATH has been set permanently for the 'httfacelond' environment."

echo "=========================================================="
echo "[6/6] Downloading required AI models..."
echo "=========================================================="
python httfacelond/utils/downloader.py

echo "=========================================================="
echo " Installation completed successfully!"
echo "=========================================================="
echo "To run the application:"
echo "  conda activate httfacelond"
echo "  python app.py"
echo "=========================================================="
