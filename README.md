# httfacelond (FP16 Accelerated)

An advanced, hardware-accelerated AI face-swapping studio for both images and videos. Built with Python, Gradio, ONNX Runtime, and OpenCV.

## 🚀 Key Features

* **High-Precision Blending**: Replaced bounding box masks with dynamic 106-point landmark-based facial convex hulls with soft feathering.
* **Foreground Occlusion Masking**: Integrated `face_occluder.onnx` to automatically detect foreground elements (hands, arms, hair, clothing) and preserve them in front of the swapped face.
* **Dynamic Auto-Thresholding**: Automatically adjusts detection confidence thresholds frame-by-frame if no face is found (ideal for tilted, profile, or sideways faces).
* **Multi-Resolution Upscaling**: Custom upscaling sizes (128px, 256px, 512px, 1024px) for face crops prior to warping, allowing crisp Ultra HD results.
* **Auto-Rotation Recovery**: Detects tilted or sideways faces by trying 90°, 180°, and 270° orientation sweeps, swapping them, and restoring the original orientation.
* **Live Frame Preview**: Upload a video, slide to any frame index, and preview the swapped face instantly before committing to a full video render.
* **Video Time Estimator**: Displays estimated render times based on frame count and execution hardware (CPU vs GPU).
* **Real-time Logs**: Dark-themed terminal UI box directly in the web app showing detailed processing stages.

---

## 🛠️ Installation & Setup

## 🛠️ Installation & Setup

### For Windows PC (venv)

1. Make sure you have [Python 3.12](https://www.python.org/downloads/) installed.
2. Double-click the **`install_windows_venv.bat`** file. This will automatically create the virtual environment, install requirements, and download the models.
3. Once completed, you can launch the studio anytime by double-clicking **`run_studio.bat`**.

### For Linux (Debian / Ubuntu - Conda)

A dedicated script is provided to automate library installation and environment building under Conda (including local CUDA configuration for GPU!).

1. Give execution permission and run the script:
   ```bash
   chmod +x install_linux_conda.sh
   ./install_linux_conda.sh
   ```
2. Once the script finishes, activate the environment and start the application:
   ```bash
   conda activate httfacelond
   python app.py
   ```

---

## 🖥️ Web Interface

Open your browser and navigate to:
* Local Link: **http://localhost:7860**

---

## 📦 Model Sources & Credits

The AI models used in this project are automatically downloaded from the following Hugging Face repositories:

*   **Face Swapper (FP16):** [inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx) (via hacksider)
*   **Face Swapper (FP32):** [inswapper_128.onnx](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx) (via ezioruan)
*   **Face Restorer (GFPGAN):** [GFPGANv1.4.onnx](https://huggingface.co/Neus/GFPGANv1.4/resolve/main/GFPGANv1.4.onnx) (via Neus)
*   **Face Occluder:** [face_occluder.onnx](https://huggingface.co/Rookiehan/facefusion/resolve/main/face_occluder.onnx) (via Rookiehan)
*   **Face Enhancer (CodeFormer):** [codeformer.onnx](https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx) (via facefusion)
*   **Face Enhancer (GPEN-BFR-512):** [GPEN-BFR-512.onnx](https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx) (via Gourieff)
*   **Face Enhancer (GPEN-BFR-1024):** [GPEN-BFR-1024.onnx](https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx) (via Gourieff)
*   **Face Detector (YOLOv11-Face):** [yolov11n-face.onnx](https://huggingface.co/AdamCodd/YOLOv11n-face-detection/resolve/main/model.onnx) (via AdamCodd)

---

## 📄 License

Copyright (c) 2026. All rights reserved. Modification, distribution, or reproduction without prior written permission is strictly prohibited. Please refer to the [LICENSE](file:///c:/Users/fds/Desktop/face/LICENSE) file for more details.



