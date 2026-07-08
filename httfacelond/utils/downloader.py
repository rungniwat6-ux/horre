import os
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[Info] Model already exists at: {dest_path}")
        return True

    print(f"[Info] Downloading model from {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print("[Info] Download completed successfully.")
        return True
    except Exception as e:
        print(f"[Error] Failed to download model: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def check_and_download_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    inswapper_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    inswapper_url = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
    
    inswapper_fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
    inswapper_fp32_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    
    gfpgan_path = os.path.join(models_dir, "GFPGANv1.4.onnx")
    gfpgan_url = "https://huggingface.co/Neus/GFPGANv1.4/resolve/main/GFPGANv1.4.onnx"
    
    occluder_path = os.path.join(models_dir, "face_occluder.onnx")
    occluder_url = "https://huggingface.co/Rookiehan/facefusion/resolve/main/face_occluder.onnx"
    
    codeformer_path = os.path.join(models_dir, "codeformer.onnx")
    codeformer_url = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx"
    
    gpen_512_path = os.path.join(models_dir, "GPEN-BFR-512.onnx")
    gpen_512_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx"
    
    gpen_1024_path = os.path.join(models_dir, "GPEN-BFR-1024.onnx")
    gpen_1024_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx"
    
    yolov11_face_path = os.path.join(models_dir, "yolov11n-face.onnx")
    yolov11_face_url = "https://huggingface.co/huygiatrng/yolov11n-face-pose-5kp/resolve/main/face_5kp_yolo11n.onnx"
    
    success1 = download_file(inswapper_url, inswapper_path)
    success2 = download_file(inswapper_fp32_url, inswapper_fp32_path)
    success3 = download_file(gfpgan_url, gfpgan_path)
    success4 = download_file(occluder_url, occluder_path)
    success5 = download_file(codeformer_url, codeformer_path)
    success6 = download_file(gpen_512_url, gpen_512_path)
    success7 = download_file(gpen_1024_url, gpen_1024_path)
    success8 = download_file(yolov11_face_url, yolov11_face_path)
    
    return success1 and success2 and success3 and success4 and success5 and success6 and success7 and success8, inswapper_path

if __name__ == "__main__":
    check_and_download_models()
