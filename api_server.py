# -*- coding: utf-8 -*-
import os
import sys
from loguru import logger
import torch
import clip
import requests
import json
import numpy as np
from flask import Flask, request, jsonify
import io

# --- Image Preprocessing Imports ---
import cv2
from PIL import Image as PILImage
from insightface.utils import face_align
from skimage.transform import estimate_transform, warp

# --- Add src to path ---
sys.path.append("./src") 

# --- Import models and config ---
try:
    # Import the main config and the config helpers
    from src.configs.config import cfg, get_cfg_defaults, update_cfg, parse_args
    from src.models.flame import FLAME
    from src.models.baselinemodels.flameparamrank_model import FlameParamRankModel
    from src.models.baselinemodels.flameparamdiffusion_model import FlameParamDiffusionModel
    from insightface.app import FaceAnalysis
    from pytorch_lightning import seed_everything
    
    # Import TesterRank
    from src.testerrank import Tester as TesterRank

except ImportError as e:
    logger.error(f"Failed to import models. Make sure './src' is in sys.path. Error: {e}")
    sys.exit(1)


# --- MICA HTTP Wrapper ---
class MICAHTTPWrapper:
    def __init__(self, api_url='http://localhost:5010'):
        self.api_url = api_url
        self._check_health()
    
    def _check_health(self):
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("MICA API server is healthy")
            else:
                raise RuntimeError("MICA API server not healthy")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("MICA API server not reachable")
    
    def get_flame_params(self, image, arcface):
        # ... (MICA helper function code) ...
        data = {
            'image': image.cpu().numpy().tolist() if torch.is_tensor(image) else image.tolist(),
            'arcface': arcface.cpu().numpy().tolist() if torch.is_tensor(arcface) else arcface.tolist()
        }
        response = requests.post(f"{self.api_url}/infer", json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                return torch.tensor(result['flame_params']).float()
            else:
                raise RuntimeError(f"MICA inference failed: {result['error']}")
        else:
            raise RuntimeError(f"HTTP request failed: {response.status_code}")

# --- Global Variables for Models ---
device = None
main_model = None       # Corresponds to `model1`
rank_model = None       # Corresponds to `TesterRank(model_rank)`
farl_model = None
farl_preprocess = None
flame_model = None
face_app = None
mica_model = None
flame_faces = None

# --- Global Constants for Preprocessing ---
input_mean = 127.5
input_std = 127.5
ofer_weight = 1.0
mica_weight = 0.0

def load_all_models(args):
    """
    Initializes all models globally, following the exact config
    logic from test_amiya.py
    """
    global device, main_model, rank_model, farl_model, farl_preprocess, flame_model, face_app, mica_model, flame_faces, ofer_weight, mica_weight
    
    logger.info("--- [SERVER] Initializing all models ---")
    
    # Set seed (from test_amiya.py)
    seed_everything(1)

    # 1. Set Device
    device = torch.device("cpu")
    logger.info(f"[SERVER] Using device: {device}")

    # 2. MICA Wrapper
    mica_api_url = cfg.get('mica_api_url', 'http://localhost:5010') 
    if mica_api_url:
        try:
            mica_model = MICAHTTPWrapper(mica_api_url)
        except Exception as e:
            logger.warning(f"[SERVER] Failed to initialize MICA: {e}")
            mica_model = None
    
    # 3. Fusion Weights
    ofer_weight = 1.0
    mica_weight = 0.0
    logger.info(f"[SERVER] Fusion weights - OFER: {ofer_weight}, MICA: {mica_weight}")

    # 4. FaRL Model
    try:
        pretrainedfarlmodel = cfg.model.pretrained
        farl_model, farl_preprocess = clip.load("ViT-B/16", device="cpu")
        farl_state_path = os.path.join(pretrainedfarlmodel, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
        farl_state = torch.load(farl_state_path, map_location=torch.device('cpu'))
        farl_model.load_state_dict(farl_state["state_dict"], strict=False)
        farl_model = farl_model.to(device)
        logger.info("[SERVER] FaRL model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load FaRL model: {e}")
        raise e

    # 5. FLAME Model
    try:
        flame_model = FLAME(cfg.model).to(device)
        flame_faces = flame_model.faces_tensor.cpu()
        logger.info("[SERVER] FLAME model loaded.")
    except Exception as e:
        logger.error(f"Failed to load FLAME model: {e}")
        raise e
        
    # 6. FaceAnalysis / InsightFace
    try:
        face_app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(224, 224), det_thresh=0.4)
        logger.info("[SERVER] InsightFace analysis model loaded.")
    except Exception as e:
        logger.error(f"Failed to load InsightFace model: {e}")
        raise e
        
    # 7. Rank Model (Wrapped in TesterRank)
    try:
        logger.info("[SERVER] Loading rank model...")
        
        # Load the base config file
        cfg_file = './src/configs/config_flameparamdiffusion_flame20.yml' 
        cfg_rank = get_cfg_defaults()
        if cfg_file is not None:
            cfg_rank = update_cfg(cfg_rank, cfg_file)


        if not hasattr(args, 'checkpoint1'):
             logger.error("Command-line args are missing '--checkpoint1'. Cannot load rank model.")
             raise ValueError("Missing args.checkpoint1")
        
        cfg_rank.train.resume_checkpoint = args.checkpoint1
        cfg_rank.model.sampling = 'ddim'
        cfg_rank.net.arch = 'archv4'
        cfg_rank.varsched.num_steps = 1000
        cfg_rank.varsched.beta_1 = 1e-4
        cfg_rank.varsched.beta_T = 1e-2
        cfg_rank.train.resume=True
        cfg_rank.train.resumepretrain = False
        cfg_rank.model.expencoder = 'arcfarl'
        cfg_rank.model.preexpencoder = 'arcface'
        cfg_rank.model.prenettype = 'preattn'
        cfg_rank.model.numsamples = 10
        cfg_rank.model.usenewfaceindex = True
        cfg_rank.model.istrial = False
        cfg_rank.net.losstype = 'Softmaxlistnetloss'
        cfg_rank.net.numattn = 1
        cfg_rank.net.predims = [300,50,10] 
        cfg_rank.model.flametype = 'flame20'
        cfg_rank.dataset.flametype = 'flame20'
        cfg_rank.model.nettype = 'listnet'
        cfg_rank.net.rankarch = 'scorecb1listnet'
        cfg_rank.net.shape_dim = 5355
        cfg_rank.net.context_dim = 1024
        cfg_rank.model.testing = True
  

        # 1. Create the internal model
        model_rank_internal = FlameParamRankModel(cfg_rank, device)
        model_rank_internal.eval()

        # 2. Wrap it in the TesterRank class
        rank_model = TesterRank(model_rank_internal, cfg_rank, device)
        
        # 3. Use the TesterRank's loading method (loads checkpoint from cfg_rank)
        rank_model.model.load_model()
        
        logger.info("[SERVER] Rank model (wrapped in TesterRank) loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load rank model: {e}")
        raise e

    # 8. Main Diffusion Model (model1 in tester)
    try:
        logger.info("[SERVER] Loading main diffusion model (for encode/decode)...")
        
        # Clone the main server config
        cfg1 = cfg.clone()
        if not hasattr(args, 'checkpoint2'):
             logger.error("Command-line args are missing '--checkpoint2'. Cannot load main model.")
             raise ValueError("Missing args.checkpoint2")

        # ...config
        
        cfg1.train.resume_checkpoint = args.checkpoint2

        cfg1.model.sampling = 'ddim'
        cfg1.model.with_exp = False
        cfg1.model.expencoder = 'arcface'
        cfg1.net.flame_dim = 300
        cfg1.net.arch = 'archv4'
        cfg1.net.context_dim = 512
        cfg1.model.nettype = 'preattn'
        cfg1.net.dims = [300,50,10]
        cfg1.net.numattn = 1
        cfg1.train.resume = True
        cfg1.dataset.flametype = 'flame20'
        cfg1.model.flametype = 'flame20'
        cfg1.model.testing = True
        main_model = FlameParamDiffusionModel(cfg1, device)
        main_model.eval()
        
        logger.info("[SERVER] Main diffusion model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load main diffusion model: {e}")
        raise e

    logger.info("--- [SERVER] All models loaded successfully ---")


# --- Flask App Definition ---
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models_loaded": main_model is not None and rank_model is not None}), 200


@app.route('/infer', methods=['POST'])
def infer():
    """
    This endpoint receives a raw image file, performs all
    preprocessing, and then runs inference.
    """
    global input_mean, input_std 
    
    try:
        # 1. --- RECEIVE AND LOAD IMAGE ---
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No 'image' file part in request"}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        pil_image = PILImage.open(io.BytesIO(image_bytes)).convert('RGB')
        cv2_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if cv2_image is None:
            return jsonify({"success": False, "error": "Failed to decode image with cv2"}), 400
        orig_image_np = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # 2. --- RUN PREPROCESSING (from SimpleTester.prepare_image) ---
        
        # 2a. Detect Face
        bboxes, kpss = face_app.det_model.detect(cv2_image, max_num=1, metric='default')
        if bboxes.shape[0] == 0:
            return jsonify({"success": False, "error": "No face detected in image"}), 400
        
        kps = kpss[0]
        bbox = bboxes[0, 0:4]

        # 2b. Generate `arcface` tensor (112x112)
        arcface_img = face_align.norm_crop(cv2_image, landmark=kps)
        blob = cv2.dnn.blobFromImages([arcface_img], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
        arcface = torch.tensor(blob[0]).float().to(device) 

        # 2c. Generate `imagefarl` tensor (224x224)
        imagefarl = farl_preprocess(pil_image).to(device)

        # 2d. Generate `normtransimage` tensor (224x224 cropped)
        scale = 1.6
        left, top, right, bottom = bbox
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * scale)
        crop_size = 224
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - 1]])
        DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        image_np_0_1 = orig_image_np / 255.
        dst_image = warp(image_np_0_1, tform.inverse, output_shape=(crop_size, crop_size))
        normtransimage = torch.tensor(dst_image.transpose(2, 0, 1)).float().to(device)

        logger.info("Preprocessing complete. Generated all 3 tensors.")

        # 3. --- RUN INFERENCE (from SimpleTester.decode) ---
        numface = 10 # Match the tiling
        with torch.no_grad():
            # Tile inputs
            arcface1 = arcface.unsqueeze(0).tile(numface,1,1,1)
            img_tensor1 = normtransimage.unsqueeze(0).tile(numface,1,1,1)
            imgfarl_tensor1 = imagefarl.unsqueeze(0).tile(numface,1,1,1)
            
            # Encode/Decode with main model
            codedict1 = main_model.encode(img_tensor1, arcface1, imgfarl_tensor1)
            opdict1 = main_model.decode(codedict1, 0, withpose=False)
            
            if 'pred_mesh' not in opdict1:
                 return jsonify({"success": False, "error": "Main model did not predict 'pred_mesh'"}), 500
            
            pred_shape_meshes = opdict1['pred_mesh']

            # --- MICA Fusion (optional) ---
            if mica_model:
                try:
                    mica_flame_params = mica_model.get_flame_params(normtransimage.unsqueeze(0), arcface.unsqueeze(0))
                    mica_flame_params = mica_flame_params.tile(numface, 1, 1)
                    logger.info("MICA params obtained, fusion would happen here.")
                except Exception as e:
                    logger.warning(f"MICA inference failed during fusion: {e}")

            # --- Ranking ---
            arcface_rank = arcface.clone()
            maxindex, sortindex = rank_model.getmaxsampleindex(arcface_rank, normtransimage, imagefarl, pred_shape_meshes)
            
            best_mesh_idx = maxindex[0] # Get the top-ranked index
            best_mesh_vertices = pred_shape_meshes[best_mesh_idx].cpu().numpy() * 1000.0
            
        # 4. --- RETURN RESULT ---
        return jsonify({
            "success": True,
            "vertices": best_mesh_vertices.tolist(),
            "faces": flame_faces.numpy().tolist()
        })

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# --- Main execution ---
if __name__ == '__main__':
    # 1. Parse command-line arguments and update the main 'cfg'
    cfg, args = parse_args()
    
    # 2. Load all models on startup, passing the args
    load_all_models(args)
    
    # 3. Run the server
    logger.info("Starting Flask server on http://0.0.0.0:5011")
    app.run(host='0.0.0.0', port=5011, debug=False)