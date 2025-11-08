# -*- coding: utf-8 -*-

import os
import sys
from glob import glob
import cv2
from PIL import Image as PILImage
import numpy as np
import torch
import re
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
import trimesh

# Make sure src is in the path
sys.path.append("./src")

from src.configs.config import cfg
from src.utils import util
from src.models.flame import FLAME

# Input normalization constants from tester_amiya.py
input_mean = 127.5
input_std = 127.5

class SimpleTester(object):
    def __init__(self, models, config=None, cfgs=None, device=None, args=None, rankmodel=None, mica_api_url=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.cfgs = cfgs
        self.args = args
        self.rankmodel = rankmodel

        # Use CPU as specified in your test_amiya.py
        self.device = torch.device("cpu")
        print(f"[INFO] Using device: {self.device}")
        
        # --- MICA HTTP Wrapper Initialization ---
        self.mica_model = None
        if mica_api_url:
            try:
                self.mica_model = MICAHTTPWrapper(mica_api_url)
                logger.info("[TESTER] MICA HTTP wrapper initialized successfully")
            except Exception as e:
                logger.warning(f"[TESTER] Failed to initialize MICA: {e}")
        
        # Fusion weights from tester_amiya.py
        self.ofer_weight = 1.0
        self.mica_weight = 0
        logger.info(f"[TESTER] Fusion weights - OFER: {self.ofer_weight}, MICA: {self.mica_weight}")

        # --- FaRL Model Initialization ---
        try:
            import clip
            self.pretrainedfarlmodel = self.cfg.model.pretrained
            self.farlmodel, self.farl_preprocess = clip.load("ViT-B/16", device="cpu")
            farl_state_path = os.path.join(self.pretrainedfarlmodel, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
            farl_state = torch.load(farl_state_path, map_location=torch.device('cpu'))
            self.farlmodel.load_state_dict(farl_state["state_dict"], strict=False)
            self.farlmodel = self.farlmodel.to(self.device)
            logger.info("FaRL model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FaRL model: {e}")
            raise e

        # --- Main Model Initialization ---
        self.model = {i:None for i in range(len(models))}
        for i in range(len(models)):
            self.model[i] = models[i].to(self.device)
            self.model[i].testing = True
            self.model[i].eval()
        
        # --- FLAME and Face Analysis Initialization ---
        flameModel = FLAME(self.cfg.model).to(self.device)
        self.faces = flameModel.faces_tensor.cpu()
        
        # Initialize InsightFace FaceAnalysis for detection and alignment
        self.app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(224, 224), det_thresh=0.4)
        logger.info("InsightFace analysis model loaded.")

    def process_folder(self):
        """
        Processes all images in the folder specified by args.imagepath.
        """
        input_dir = self.args.imagepath
        output_dir = self.args.outputpath
        
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processing images from: {input_dir}")
        logger.info(f"Saving results to: {output_dir}")

        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(input_dir, ext)))

        for image_path in tqdm(image_paths, desc="Processing images"):
            logger.info(f"--- Processing: {image_path} ---")
            try:
                # 1. Prepare all inputs from a single image path
                prepared_data = self.prepare_image(image_path)
                
                if prepared_data is None:
                    logger.warning(f"Skipping {image_path}, no face detected or error.")
                    continue
                
                # 2. Build the input dictionary for the decode function
                image_name = os.path.basename(image_path)
                image_name_noext = os.path.splitext(image_name)[0]

                # 'actor' and 'type' are used for subfolders in decode
                # We'll set them to fixed values for a simple structure
                result = {
                    **prepared_data, # Contains all image tensors
                    'imgname': image_name,
                    'imgname_noext': image_name_noext,
                    'best_id': 'single_image_test',
                    'id': 'single_image_test',
                    'numface': 10, # Match the tile(10) in decode
                    'actor': 'results', # All results in one subfolder
                    'type': 'images',   # All results in one subfolder
                    'kpt': prepared_data['kps'],
                    'outfile': 'output' # Root folder for decode's save path
                }

                # 3. Call the decode function (copied from Tester2)
                self.decode(result)

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                import traceback
                traceback.print_exc()

    def prepare_image(self, image_path):
        """
        Loads a single image, detects the face, and prepares all
        necessary tensors (ArcFace, FaRL, and main diffusion input).
        """
        try:
            # Load image with PIL (for FaRL) and cv2 (for InsightFace)
            pil_image = PILImage.open(image_path).convert('RGB')
            cv2_image = cv2.imread(image_path)
            if cv2_image is None:
                raise IOError("Failed to read image with cv2.")
            # This is the original image in numpy format (H, W, C), RGB
            orig_image_np = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return None

        # 1. Detect Face
        bboxes, kpss = self.app.det_model.detect(cv2_image, max_num=1, metric='default')
        
        if bboxes.shape[0] == 0:
            logger.warning(f"No face detected in {image_path}")
            return None
        
        i = 0 # Use the first (and only) detected face
        bbox = bboxes[i, 0:4]
        kps = kpss[i]
        face = Face(bbox=bbox, kps=kps)

        # 2. Get ArcFace Input (112x112)
        # Logic from Tester2.process_image
        arcface_img = face_align.norm_crop(cv2_image, landmark=face.kps)
        blob = cv2.dnn.blobFromImages([arcface_img], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
        arcface_tensor = torch.tensor(blob[0]).float().to(self.device) # Shape: (3, 112, 112)

        # 3. Get FaRL Input (224x224)
        # Logic from Tester2.test_now
        imagefarl_tensor = self.farl_preprocess(pil_image).to(self.device) # Shape: (3, 224, 224)

        # 4. Get Main Model Input (224x224, cropped)
        # Logic from Tester2.process_folder, but using DETECTED bbox
        scale = 1.6 # Use the same scale as NoW
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

        # Transpose to (C, H, W)
        normtransimage = dst_image.transpose(2, 0, 1)
        normtransimage_tensor = torch.tensor(normtransimage).float().to(self.device) # Shape: (3, 224, 224)

        return {
            'origimage': orig_image_np, # (H, W, C) numpy
            'normimage': dst_image, # (224, 224, C) numpy [0,1]
            'normtransimage': normtransimage_tensor, # (3, 224, 224) tensor
            'arcface': arcface_tensor, # (3, 112, 112) tensor
            'imagefarl': imagefarl_tensor, # (3, 224, 224) tensor
            'kps': kps # numpy array
        }


    # --- Copied verbatim from tester_amiya.py ---

    def decode(self, input):
        print("in decode", flush=True)
        origimage = input['origimage']
        normimage = input['normimage']
        normtransimage = input['normtransimage']
        if 'uncutimage' in input:
            uncutimage = input['uncutimage']
        arcface = input['arcface']
        imagefarl = input['imagefarl']
        image_name = input['imgname']
        image_name_noext = input['imgname_noext']
        best_id = input['best_id']
        id = input['id']
        numface= input['numface'] # This is 10 from prepare_image
        outfile= input['outfile']
        actor=input['actor']
        type=input['type']
        kpt=input['kpt']
        istest='val'

        interpolate = 224
        arcface_rank = arcface.clone()
        with torch.no_grad():
            
            # Tile inputs to match numface (e.g., 10)
            arcface1 = arcface.unsqueeze(0).tile(numface,1,1,1)
            img_tensor1 = normtransimage.unsqueeze(0).tile(numface,1,1,1).to(self.device)
            imgfarl_tensor1 = imagefarl.unsqueeze(0).tile(numface,1,1,1).to(self.device)

            codedict1 = self.model[0].encode(img_tensor1, arcface1, imgfarl_tensor1)
            opdict1 = self.model[0].decode(codedict1, 0, withpose=False)
            pred_flameparam1 = opdict1['pred_flameparam']
            if 'pred_mesh' in opdict1:
                pred_shape_meshes = opdict1['pred_mesh']
                pred_shape_lmk = self.model[0].flame.compute_landmarks(pred_shape_meshes)
                print(pred_shape_lmk.shape," lmk sahpe")
            
            ############################################################################
            #MICA
            mica_flame_params = None
            if self.mica_model is not None:
                try:
                    # Use single image for MICA (not tiled)
                    mica_images = normtransimage.unsqueeze(0)
                    mica_arcface = arcface.unsqueeze(0)
                    
                    mica_flame_params = self.mica_model.get_flame_params(mica_images, mica_arcface)
                    mica_flame_params = mica_flame_params.tile(numface, 1, 1)  # Match OFER batch size
                    
                except Exception as e:
                    logger.warning(f"MICA inference failed: {e}")

            ############################################################################
            # try fusing
            if mica_flame_params is not None and 'pred_mesh' in opdict1:
                pred_shape_meshes = self.fuse_flame_params(pred_shape_meshes,mica_flame_params)
                pred_shape_lmk = self.model[0].flame.compute_landmarks(pred_shape_meshes)
                print(pred_shape_lmk.shape," lmk sahpe")
                logger.info(f"Fused FLAME params - OFER:{self.ofer_weight}, MICA:{self.mica_weight}")
            elif 'pred_mesh' not in opdict1:
                 logger.warning("No 'pred_mesh' in opdict1, skipping fusion.")
                 return # Can't continue without meshes

            ############################################################################
            shape = pred_flameparam1[:,:300]
            print("num shape = ", shape.shape, flush=True)
            print("num shape = ", pred_shape_meshes.shape, flush=True)
            ######### GET BEST RANK #######################
            maxindex, sortindex = self.rankmodel.getmaxsampleindex(arcface_rank, normtransimage, imagefarl, pred_shape_meshes)
            opdict1 = self.model[0].decode(codedict1, 0, withpose=False)

            # (All commented out code from original decode is omitted here for clarity)

        os.makedirs(os.path.join(self.cfg.output_dir, f'{outfile}', 'shapesample'), exist_ok=True)
        shape_dst_folder = os.path.join(self.cfg.output_dir, f'{outfile}', 'shapesample', actor, type)
        os.makedirs(shape_dst_folder, exist_ok=True)

        image_name = re.sub('arcface_input/','',image_name)
        a = image_name
        savepath = os.path.split(os.path.join(shape_dst_folder, a))[0]
        os.makedirs(savepath, exist_ok=True)

        # Save the best ranked mesh
        for num in range(1): # Save only the top 1
            currname = a[:-4]+'.jpg'
            saveshapepath = os.path.join(shape_dst_folder, currname.replace('jpg', 'obj'))
            
            best_mesh_idx = maxindex
            trimesh.Trimesh(vertices=pred_shape_meshes[maxindex[num]].cpu() * 1000.0, faces=self.faces, process=False).export(saveshapepath)
            logger.info(f"Saved mesh to {saveshapepath}")
            
            lmk = pred_shape_lmk[best_mesh_idx]
            landmark_51_best = lmk[0, 17:]
            landmark_7_best = landmark_51_best[[19, 22, 25, 28, 16, 31, 37]]
            saveshapepath_npy = os.path.join(shape_dst_folder, currname.replace('.jpg', ''))
            np.save(f'{saveshapepath_npy}', landmark_7_best.cpu().numpy() * 1000.0)
            logger.info(f"Saved landmarks to {saveshapepath_npy}.npy")

    def fuse_flame_params(self, ofer_params, mica_params):
        """
        Fuse FLAME parameters from OFER and MICA models
        (Copied verbatim from tester_amiya.py)
        """
        # Ensure both have same shape
        ofer_dim = ofer_params.shape[1]
        mica_dim = mica_params.shape[1]
        
        logger.info(f"Fusing FLAME params - OFER dim: {ofer_dim}, MICA dim: {mica_dim}")
        
        if ofer_dim != mica_dim:
            min_dim = min(ofer_dim, mica_dim)
            ofer_params = ofer_params[:, :min_dim]
            mica_params = mica_params[:, :min_dim]
            logger.warning(f"FLAME parameter dimension mismatch. Using first {min_dim} dimensions.")
        
        # Weighted fusion
        fused_params = (self.ofer_weight * ofer_params + 
                       self.mica_weight * mica_params)
        
        logger.info(f"Fusion completed - output shape: {fused_params.shape}")
        
        return fused_params


# --- MICA HTTP Wrapper Class (Copied verbatim from tester_amiya.py) ---
import requests
import json

class MICAHTTPWrapper:
    def __init__(self, api_url='http://localhost:5010'):
        self.api_url = api_url
        self._check_health()
    
    def _check_health(self):
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print("MICA API server is healthy")
            else:
                raise RuntimeError("MICA API server not healthy")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("MICA API server not reachable")
    
    def get_flame_params(self, image, arcface):
        """Get FLAME parameters via HTTP API"""
        data = {
            'image': image.cpu().numpy().tolist() if torch.is_tensor(image) else image.tolist(),
            'arcface': arcface.cpu().numpy().tolist() if torch.is_tensor(arcface) else arcface.tolist()
        }
        
        response = requests.post(
            f"{self.api_url}/infer",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                return torch.tensor(result['flame_params']).float()
            else:
                raise RuntimeError(f"MICA inference failed: {result['error']}")
        else:
            raise RuntimeError(f"HTTP request failed: {response.status_code}")