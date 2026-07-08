import os
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

class GPUPipeline:
    def __init__(self, gpu_id, providers, inswapper_path, gfpgan_path, occluder_path):
        self.gpu_id = gpu_id
        self.providers = []
        
        # Build specific providers list with target device_id
        for p in providers:
            p_name = p[0] if isinstance(p, tuple) else p
            p_opts = p[1] if isinstance(p, tuple) and len(p) >= 2 else {}
            
            if p_name.lower() == 'cudaexecutionprovider':
                opts = dict(p_opts)
                opts['device_id'] = int(gpu_id)
                self.providers.append((p_name, opts))
            else:
                self.providers.append(p)
                
        print(f"[GPUPipeline {gpu_id}] Initializing sessions with providers: {self.providers}")
        
        self.target_app = FaceAnalysis(name='buffalo_l', providers=self.providers, allowed_modules=['detection', 'landmark_2d_106'])
        self.target_app.prepare(ctx_id=int(gpu_id), det_size=(640, 640))
        
        self.swapper = insightface.model_zoo.get_model(inswapper_path, providers=self.providers)
        
        if gfpgan_path and os.path.exists(gfpgan_path):
            self.gfpgan_session = ort.InferenceSession(gfpgan_path, providers=self.providers)
        else:
            self.gfpgan_session = None
            
        if occluder_path and os.path.exists(occluder_path):
            self.occluder_session = ort.InferenceSession(occluder_path, providers=self.providers)
        else:
            self.occluder_session = None


class CPUPipeline:
    def __init__(self, inswapper_path, gfpgan_path, occluder_path):
        self.gpu_id = -1
        self.providers = ['CPUExecutionProvider']
        
        print(f"[CPUPipeline] Initializing sessions with CPU provider")
        
        self.target_app = FaceAnalysis(name='buffalo_l', providers=self.providers, allowed_modules=['detection', 'landmark_2d_106'])
        self.target_app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.swapper = insightface.model_zoo.get_model(inswapper_path, providers=self.providers)
        
        if gfpgan_path and os.path.exists(gfpgan_path):
            self.gfpgan_session = ort.InferenceSession(gfpgan_path, providers=self.providers)
        else:
            self.gfpgan_session = None
            
        if occluder_path and os.path.exists(occluder_path):
            self.occluder_session = ort.InferenceSession(occluder_path, providers=self.providers)
        else:
            self.occluder_session = None
