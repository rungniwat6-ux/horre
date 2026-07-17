# httfacelond

Hardware-accelerated AI face swap studio for images and videos. Built with Python, Gradio, ONNX Runtime, InsightFace, MediaPipe, and OpenCV.

## ระบบที่มีในโปรแกรม

### Image Swap

- สลับหน้าในภาพนิ่งจาก source face ไปยัง target image
- รองรับ batch image swap สำหรับหลายภาพพร้อมกัน
- อัปโหลด target images แบบทั้งโฟลเดอร์ผ่านหน้าเว็บได้
- ใส่ local folder path โดยตรงได้ เช่น `C:\Users\fds\Pictures\targets`
- บันทึกผล batch ลงโฟลเดอร์อัตโนมัติ และสร้างไฟล์ `swapped_batch_results.zip`
- แสดงผล batch ผ่าน gallery ในหน้าเว็บ
- รองรับไฟล์ภาพ `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`

### Video Swap

- สลับหน้าในวิดีโอจาก source face image ไปยัง target video
- เลือก frame เพื่อ preview ก่อน render เต็มได้
- แสดง estimated processing time ตามจำนวน frame และโหมด CPU/GPU
- แสดง real-time progress, speed, ETA และ logs ระหว่างประมวลผล
- รองรับ frame step/skip สำหรับลดเวลาประมวลผล
- รองรับ threaded execution และเลือกจำนวน concurrent workers ได้
- รวมเสียงเดิมกลับเข้า output video หลังประมวลผล

### Face Detection และ Masking

- ใช้ InsightFace SCRFD เป็น detector หลัก
- มี YOLOv11-Face เป็นตัวเลือก detector สำรอง
- Auto-threshold fallback เมื่อหาใบหน้าไม่เจอใน threshold แรก
- Auto-rotation recovery สำหรับภาพหรือวิดีโอที่ใบหน้าเอียง/หมุน
- รองรับ InsightFace 106-point landmark mask
- รองรับ MediaPipe FaceMesh 468-point mask
- มี soft feathering เพื่อลดขอบแข็งบริเวณใบหน้า

### Realism และ Enhancement

- Face restorer/enhancer:
  - GFPGANv1.4
  - CodeFormer
  - GPEN-BFR-512
  - GPEN-BFR-1024
- ปรับ enhance strength ได้
- Match lighting และ skin tone ด้วย color transfer
- Match face shape aspect ratio
- ปรับ face swap blend strength เพื่อคุมความเหมือน/ความเนียน
- ปรับ face crop upscale resolution ได้ตั้งแต่ 128 ถึง 2048
- Occlusion masking เพื่อเก็บวัตถุด้านหน้า เช่น มือ ผม แขน หรือเสื้อผ้า

### Hardware และ Models

- รองรับ GPU mode ผ่าน ONNX Runtime CUDA ถ้ามี GPU ที่ใช้งานได้
- รองรับ CPU-only mode
- เลือก GPU device ได้เมื่อมีหลาย GPU
- เลือก swapper model ได้:
  - `inswapper_128_fp16.onnx`
  - `inswapper_128.onnx`
- มีปุ่ม clear VRAM เพื่อ unload models ระหว่างใช้งาน

## Installation & Setup

### Windows PC (venv)

1. ติดตั้ง Python 3.12 และเพิ่ม Python เข้า PATH
2. ดับเบิลคลิก `install_windows_venv.bat`
3. หลังติดตั้งเสร็จ เปิดโปรแกรมด้วย `run_studio.bat`

### Linux (Debian / Ubuntu - Conda)

```bash
chmod +x install_linux_conda.sh
./install_linux_conda.sh
conda activate httfacelond
python app.py
```

## Web Interface

หลังเปิดโปรแกรม เข้าใช้งานที่:

```text
http://127.0.0.1:7860
```

หรือ:

```text
http://localhost:7860
```

## วิธีใช้แบบย่อ

### ใช้ Image Swap ภาพเดียว

1. อัปโหลด `Source Face`
2. อัปโหลด `Target Image`
3. เลือก model/settings ตามต้องการ
4. กด `Start Face Swap`

### ใช้ Batch Image Swap ทั้งโฟลเดอร์

1. อัปโหลด `Source Face`
2. อัปโหลดโฟลเดอร์ในช่อง `Batch Target Images / Folder Upload`
3. หรือใส่ path ในช่อง `Batch Target Local Folder Path`
4. กด `Start Batch Folder Swap`
5. ผลลัพธ์จะอยู่ใน gallery, output folder และ zip file

### ใช้ Video Swap

1. อัปโหลด `Source Face Image`
2. อัปโหลด `Target Video`
3. เลือก frame แล้วกด preview ถ้าต้องการตรวจหน้าก่อน
4. กด `Start Face Swap Video`

## Model Sources & Credits

The AI models are automatically downloaded from these sources:

- Face Swapper FP16: [inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)
- Face Swapper FP32: [inswapper_128.onnx](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx)
- Face Restorer GFPGAN: [GFPGANv1.4.onnx](https://huggingface.co/Neus/GFPGANv1.4/resolve/main/GFPGANv1.4.onnx)
- Face Occluder: [face_occluder.onnx](https://huggingface.co/Rookiehan/facefusion/resolve/main/face_occluder.onnx)
- CodeFormer: [codeformer.onnx](https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx)
- GPEN-BFR-512: [GPEN-BFR-512.onnx](https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx)
- GPEN-BFR-1024: [GPEN-BFR-1024.onnx](https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx)
- YOLOv11-Face: [yolov11n-face.onnx](https://huggingface.co/AdamCodd/YOLOv11n-face-detection/resolve/main/model.onnx)

## License

Copyright (c) 2026. All rights reserved. Modification, distribution, or reproduction without prior written permission is strictly prohibited.
