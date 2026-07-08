import sys
import os

print("==========================================================")
print("             ONNX Runtime GPU Diagnoser")
print("==========================================================")

# 1. Check Python version
print(f"[1/4] Python Version: {sys.version}")

# 2. Check ONNX Runtime installation
try:
    import onnxruntime as ort
    print(f"[2/4] ONNX Runtime package found (version: {ort.__version__})")
    available = ort.get_available_providers()
    print(f"      Available Providers in ORT: {available}")
    
    # Check if GPU providers are active
    if 'TensorRTExecutionProvider' in available:
        print("      ✅ TensorRT is available!")
    if 'CUDAExecutionProvider' in available:
        print("      ✅ CUDA is available!")
    else:
        print("      ❌ WARNING: CUDAExecutionProvider is NOT available in ONNX Runtime!")
        print("         Your ONNX Runtime package is likely CPU-only.")
except ImportError:
    print("      ❌ ONNX Runtime is NOT installed!")

# 3. Check environment variables
print("[3/4] Environment Variables:")
print(f"      LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
print(f"      CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not Set')}")

# 4. Check system CUDA libraries via nvcc or find
print("[4/4] Hardware & System CUDA info:")
import subprocess
try:
    nvcc = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
    print(f"      nvcc compiler version:\n{nvcc}")
except Exception:
    print("      ❌ nvcc command not found (normal if runtime-only container)")

try:
    smi = subprocess.check_output(["nvidia-smi"]).decode("utf-8").split("\n")[2:10]
    print("      nvidia-smi output:")
    for line in smi:
        print(f"         {line.strip()}")
except Exception:
    print("      ❌ nvidia-smi command not found (GPU drivers might not be loaded!)")

print("\n==========================================================")
print("                  Actionable Solution:")
print("==========================================================")
if 'ort' in locals() and 'CUDAExecutionProvider' not in available:
    print("Your environment is using the CPU version of ONNX Runtime.")
    print("Please run these exact commands in your terminal to enable GPU support:")
    print("\n  pip uninstall -y onnxruntime onnxruntime-gpu")
    print("  pip install onnxruntime-gpu")
    print("  pip install nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12")
    print("\n  # Then, set the paths in your shell:")
    print("  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH")
else:
    print("✅ ONNX Runtime GPU is configured correctly! If it is still slow, check if 'CUDA' is selected in the Web UI hardware dropdown.")
print("==========================================================")
