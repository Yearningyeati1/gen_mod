# -*- coding: utf-8 -*-
# thanks to OFER codebase
# tester class for NOW evaluation

import os
import sys
from glob import glob

import cv2
from PIL import Image as PILImage
import numpy as np
import torch
import re
import torch.nn.functional as F
import torch.distributed as dist
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
#from pytorch3d.io import save_ply
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

from src.configs.config import cfg
from src.utils import util
from src.models.flame import FLAME
import trimesh
import scipy.io

sys.path.append("./src")
input_mean = 127.5
input_std = 127.5

NOW_VALIDATION = '/Users/amiyachowdhury/Desktop/now_evaluation/datadir/NoW_Evaluation/dataset/imagepathsvalidation.txt'
NOW_TEST = '/work/pselvaraju_umass_edu/Project_FaceDiffusion/FACEDATA/NOW/imagepathstest.txt'


class Tester2(object):
    #def __init__(self, model1, model2, model3, config=None, device=None, args=None):
    def __init__(self, models, config=None, cfgs=None, device=None, args=None, rankmodel=None, mica_api_url=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.cfgs = cfgs

        # if torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # elif torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # else:
        #     self.device = torch.device("cpu")
        self.device = torch.device("cpu")
        print(f"[INFO] Using device: {self.device}")
#################################################################################
        # Initialize MICA wrapper
        self.mica_model = None
        if mica_api_url:
            try:
                self.mica_model = MICAHTTPWrapper(mica_api_url)
                logger.info("[TESTER] MICA HTTP wrapper initialized successfully")
            except Exception as e:
                logger.warning(f"[TESTER] Failed to initialize MICA: {e}")
        
        
        self.mica_weight = 0
        self.ofer_weight = 1 - self.mica_weight
        logger.info(f"[TESTER] Fusion weights - OFER: {self.ofer_weight}, MICA: {self.mica_weight}")
    
################################################################################
        #self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.n_images = self.cfg.dataset.n_images
        self.render_mesh = True
        self.embeddings = {}
        self.nowimages = self.cfg.test.now_images
        self.affectnetimages = self.cfg.test.affectnet_images
        self.aflw2000images = self.cfg.test.aflw2000_images
        self.args = args
        self.rankmodel = rankmodel

        import clip
        self.pretrainedfarlmodel = self.cfg.model.pretrained
        self.farlmodel, self.farl_preprocess = clip.load("ViT-B/16", device="cpu")
        farl_state = torch.load(os.path.join(self.pretrainedfarlmodel, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth"),map_location=torch.device('cpu'))
        self.farlmodel.load_state_dict(farl_state["state_dict"], strict=False)
        self.farlmodel = self.farlmodel.to(self.device)

        self.model = {i:None for i in range(len(models))}
        for i in range(len(models)):
            self.model[i] = models[i].to(self.device)
            self.model[i].testing = True
            self.model[i].eval()

        flameModel = FLAME(self.cfg.model).to(self.device)
        self.faces = flameModel.faces_tensor.cpu()


    def load_checkpoint(self, model, ckpt_path):
        #dist.barrier()
        #map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        map_location = torch.device("cpu")
        checkpoint = torch.load(ckpt_path, map_location)
        print(self.model.net)

        if 'arcface' in checkpoint:
            print("arcface")
            model.arcface.load_state_dict(checkpoint['arcface'])
        if 'farl' in checkpoint:
            print("farl")
            model.farlmodel.load_state_dict(checkpoint['farl'])
            model.arcface.load_state_dict(checkpoint['arcface'])
        if 'hseencoder' in checkpoint:
            print("hseencoder")
            model.hseencoder.load_state_dict(checkpoint['hseencoder'])
        if 'resnet' in checkpoint:
            print("resnet")
            model.resnet.load_state_dict(checkpoint['resnet'])
        if 'net' in checkpoint:
            print("net")
            model.net.load_state_dict(checkpoint['net'])
        if 'fnet' in checkpoint:
            print("fnet")
            model.fnet.load_state_dict(checkpoint['fnet'])
        if 'var_sched' in checkpoint:
            print("var_sched")
            print(checkpoint['var_sched'])
            model.var_sched.load_state_dict(checkpoint['var_sched'], strict=False)
        if 'diffusion' in checkpoint:
            print("diffusion")
            model.diffusion.load_state_dict(checkpoint['diffusion'], strict=False)

        print("done", flush=True)
        logger.info(f"[TESTER] Resume from {ckpt_path}")
        return model

    def load_model_dict(self, model_dict):
        dist.barrier()

        self.model.canonicalModel.load_state_dict(model_dict['canonicalModel'])
        self.model.arcface.load_state_dict(model_dict['arcface'])

    def process_image(self, img, app):
        images = []
        bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
        if bboxes.shape[0] != 1:
            logger.error('Face not detected!')
            return images
        i = 0
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        aimg = face_align.norm_crop(img, landmark=face.kps)
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)

        images.append(torch.tensor(blob[0])[None])

        return images

    def process_folder(self, folder, app):
        images = []
        image_names = []
        arcface = []
        count = 0
        files_actor = sorted(sorted(os.listdir(folder)))
        for file in files_actor:
            if file.startswith('._'):
                continue
            image_path = folder + '/' + file
            logger.info(image_path)
            image_names.append(image_path)
            count += 1

            ### NOW CROPPING
            scale = 1.6
            # scale = np.random.rand() * (1.8 - 1.1) + 1.1
            bbx_path = image_path.replace('.jpg', '.npy').replace('iphone_pictures', 'detected_face')
            bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
            left = bbx_data['left']
            right = bbx_data['right']
            top = bbx_data['top']
            bottom = bbx_data['bottom']

            image = imread(image_path)[:, :, :3]

            h, w, _ = image.shape
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * scale)

            crop_size = 224
            # crop image
            src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
            DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = image / 255.
            dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))

            arcface += self.process_image(cv2.cvtColor(dst_image.astype(np.float32) * 255.0, cv2.COLOR_RGB2BGR), app)

            dst_image = dst_image.transpose(2, 0, 1)
            print("process folder :",dst_image.shape)
            images.append(torch.tensor(dst_image)[None])

        images = torch.cat(images, dim=0).float()
        arcface = torch.cat(arcface, dim=0).float()
        print("images = ", count)

        return images, arcface, image_names

    def get_name(self, best_model, id):
        if '_' in best_model:
            name = id if id is not None else best_model.split('_')[-1][0:-4]
        else:
            name = id if id is not None else best_model.split('/')[-1][0:-4]
        return name


    def load_shape_cfg(self, ckpt, cfg):
        self.model[0].with_exp = False
        self.model[0].expencoder = cfg.model.expencoder
        self.model[0].net.flame_dim = cfg.net.flame_dim
        self.model[0].net.expencoder = cfg.model.expencoder

    def load_cfg(self, cfg, best_model):
        self.model.var_sched.num_steps = cfg.varsched.num_steps
        self.model.var_sched.beta_1 = cfg.varsched.beta_1
        self.model.var_sched.beta_T = cfg.varsched.beta_T
        self.model.net.flame_dim = cfg.net.flame_dim
        self.model.net.arch = cfg.net.arch
        self.model.expencoder = cfg.model.expencoder
        self.model.with_exp = cfg.model.with_exp
        self.model.sampling = cfg.model.sampling

        self.model = self.load_checkpoint(best_model)
        self.model.var_sched.num_steps = cfg.varsched.num_steps
        self.model.var_sched.beta_1 = cfg.varsched.beta_1
        self.model.var_sched.beta_T = cfg.varsched.beta_T
        self.model.net.flame_dim = cfg.net.flame_dim
        self.model.net.arch = cfg.net.arch
        self.model.expencoder = cfg.model.expencoder
        self.model.with_exp = cfg.model.with_exp
        self.model.sampling = cfg.model.sampling

    def load_cfgs(self, ckpts, cfgs):
        num = len(ckpts)

        for i in range(num):
            print(i, flush=True)
            self.model[i].with_exp = cfgs[i].model.with_exp
            self.model[i].sampling = cfgs[i].model.sampling
            self.model[i].with_lmk = cfgs[i].model.with_lmk
            self.model[i].expencoder = cfgs[i].model.expencoder
            self.model[i].net.flame_dim = cfgs[i].net.flame_dim
            self.model[i].net.arch = cfgs[i].net.arch
            self.model[i].net.context_dim = cfgs[i].net.context_dim
            self.model[i].var_sched.num_steps = cfgs[i].varsched.num_steps
            self.model[i].var_sched.beta_1 = cfgs[i].varsched.beta_1
            self.model[i].var_sched.beta_T = cfgs[i].varsched.beta_T

            self.model[i] = self.load_checkpoint(self.model[i], ckpts[i])

            self.model[i].with_exp = cfgs[i].model.with_exp
            self.model[i].sampling = cfgs[i].model.sampling
            self.model[i].with_lmk = cfgs[i].model.with_lmk
            self.model[i].expencoder = cfgs[i].model.expencoder

            if i != 2:
                self.model[i].net.flame_dim = cfgs[i].net.flame_dim
                self.model[i].net.arch = cfgs[i].net.arch
                self.model[i].net.context_dim = cfgs[i].net.context_dim
                self.model[i].var_sched.num_steps = cfgs[i].varsched.num_steps
                self.model[i].var_sched.beta_1 = cfgs[i].varsched.beta_1
                self.model[i].var_sched.beta_T = cfgs[i].varsched.beta_T


    def test_now(self, name='now_test', id='nowcache', numface=1, istest='val', isocc=0):
        self.now(name, id, numface, istest, isocc)


    def save_mesh(self, file, vertices):
        scaled = vertices * 1000.0
        save_ply(file, scaled.cpu(), self.model.render.faces[0].cpu())


    def cache_to_cuda(self, cache):
        for key in cache.keys():
            i, a = cache[key]
            cache[key] = (i.to(self.device), a.to(self.device))
        return cache

    def create_now_cache(self):
        cache_path = os.path.join(self.cfg.test.cache_path, 'test_now_cache.pt')
        if os.path.exists(cache_path):
            cache = self.cache_to_cuda(torch.load(cache_path))
            return cache
        else:
            cache = {}

        app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224), det_thresh=0.4)

        for actor in tqdm(sorted(os.listdir(self.nowimages))):
            image_paths = sorted(glob(os.path.join(self.nowimages , actor , '*')))
            print(image_paths, flush=True)
            for folder in image_paths:
                print(folder)
                images, arcface, image_names = self.process_folder(folder, app)
                for i in range(len(image_names)):
                    cache[image_names[i]] = (images[i], arcface[i])
                #cache[folder] = (images, arcface)
                print(cache.keys())
                #exit()

        torch.save(cache, cache_path) 
        return self.cache_to_cuda(cache)

    def now(self, best_id, id, numface=1, istest='val', isocc=0):
        logger.info(f"[TESTER] NoW validation has begun!")
        for i in range(len(self.model.keys())):
            self.model[i].eval()

        if istest == 'val':
            valread = open(NOW_VALIDATION, 'r')
        else:
            valread = open(NOW_TEST, 'r')
        nowimages = self.nowimages
        if isocc:
            nowimages = re.sub('arcface_input', 'arcface_occluded_'+str(isocc)+'_input', nowimages)

        for line in valread:
            line = line.strip()
            print(line, flush=True)
            data = line.strip().split('/')
            actor, type, image_name = data[0], data[1], data[2]
            images = os.path.join(nowimages, line)
            imagefarl = self.farl_preprocess(PILImage.open(images))

            image_name_noext = images[:-4]
            if id == 'nowcache':
                cache = self.create_now_cache()
                img_cache = re.sub('arcface_input', 'final_release_version/iphone_pictures', images)
                origimage, arcface = cache[img_cache]
                normimage = origimage
                print(normimage)
                print("now", normimage.shape)
                normtransimage = normimage 

            else:
                print("else")
                arcfacepath = re.sub('jpg','npy', images)
                if not os.path.exists(arcfacepath):
                    print("hello")
                    continue
                print("hello")
                arcface = torch.tensor(np.load(re.sub('jpg','npy', images))).float().to(self.device)
                origimage = imread(images)
                normimage = origimage / 255.
                normtransimage = normimage.transpose(2, 0, 1)
                #origimage = origimage / 255.
                #origimage = origimage.transpose(2, 0, 1)

            result = {'origimage': origimage,
                    'normimage': normimage,
                    'normtransimage': normtransimage,
                    'arcface': arcface,
                    'imagefarl': imagefarl,
                    'imgname': image_name,
                    'imgname_noext': image_name_noext,
                    'best_id': best_id,
                    'id': id,
                    'numface': numface,
                    'actor': actor,
                    'type':type,
                    'kpt': None,
                    'img450': '',
                    'outfile': 'now'}
            self.decode(result)
        valread.close()

    #def decode(self, origimage, arcface, image_name, image_name1, best_id, id, numface=1, outfile='now', actor='', type='', istest='val'):
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
        numface= input['numface']
        outfile= input['outfile']
        actor=input['actor']
        type=input['type']
        kpt=input['kpt']
        istest='val'

        interpolate = 224
        # origimage_copy = origimage.copy()
        arcface_rank = arcface.clone()
        with torch.no_grad():
            
            #codedict1 = self.model[0].encode(torch.Tensor(normtransimage).unsqueeze(0).to('cuda'), arcface.unsqueeze(0))
            arcface1 = arcface.tile(10,1,1,1)
            img_tensor1 = torch.Tensor(normtransimage).tile(10,1,1,1).to(self.device)
            imgfarl_tensor1 = torch.Tensor(imagefarl).tile(10,1,1,1).to(self.device)
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
                    mica_images = torch.Tensor(normtransimage).unsqueeze(0)
                    mica_arcface = arcface.unsqueeze(0)
                    
                    mica_flame_params = self.mica_model.get_flame_params(mica_images, mica_arcface)
                    mica_flame_params = mica_flame_params.tile(10, 1, 1)  # Match OFER batch size
                    
                except Exception as e:
                    logger.warning(f"MICA inference failed: {e}")


            ############################################################################
            # try fusing

            if mica_flame_params is not None:
                pred_shape_meshes = self.fuse_flame_params(pred_shape_meshes,mica_flame_params)
                pred_shape_lmk = self.model[0].flame.compute_landmarks(pred_shape_meshes)
                print(pred_shape_lmk.shape," lmk sahpe")

                logger.info(f"Fused FLAME params - OFER:{self.ofer_weight}, MICA:{self.mica_weight}")

            ############################################################################
            shape = pred_flameparam1[:,:300]
            print("num shape = ", shape.shape, flush=True)
            print("num shape = ", pred_shape_meshes.shape, flush=True)
            ######### GET BEST RANK #######################
            maxindex, sortindex = self.rankmodel.getmaxsampleindex(arcface_rank, normtransimage, imagefarl, pred_shape_meshes)
            opdict1 = self.model[0].decode(codedict1, 0, withpose=False)
            print(maxindex)

        os.makedirs(os.path.join(self.cfg.output_dir, f'{outfile}', 'shapesample'), exist_ok=True)
        shape_dst_folder = os.path.join(self.cfg.output_dir, f'{outfile}', 'shapesample', actor, type)
        os.makedirs(shape_dst_folder, exist_ok=True)

        image_name = re.sub('arcface_input/','',image_name)
        a = image_name
        savepath = os.path.split(os.path.join(shape_dst_folder, a))[0]
        os.makedirs(savepath, exist_ok=True)
        for num in range(1):
            currname = a[:-4]+'.jpg'
            saveshapepath = os.path.join(shape_dst_folder, currname.replace('jpg', 'obj'))
            trimesh.Trimesh(vertices=pred_shape_meshes[maxindex[num]].cpu() * 1000.0, faces=self.faces, process=False).export(saveshapepath)
            lmk = pred_shape_lmk[maxindex]
            landmark_51_best = lmk[0, 17:]
            landmark_7_best = landmark_51_best[[19, 22, 25, 28, 16, 31, 37]]
            saveshapepath = os.path.join(shape_dst_folder, currname.replace('.jpg', ''))
            np.save(f'{saveshapepath}', landmark_7_best.cpu().numpy() * 1000.0)


    def fuse_flame_params(self, ofer_params, mica_params):
        """
        Fuse FLAME parameters from OFER and MICA models
        
        Args:
            ofer_params: FLAME parameters from OFER model [batch_size, shape_dim]
            mica_params: FLAME parameters from MICA model [batch_size, shape_dim]
        
        Returns:
            fused_params: Weighted combination of both parameters
        """
        # Ensure both have same shape
        ofer_dim = ofer_params.shape[1]
        mica_dim = mica_params.shape[1]
        
        logger.info(f"Fusing FLAME params - OFER dim: {ofer_dim}, MICA dim: {mica_dim}")
        
        if ofer_dim != mica_dim:
            # Use minimum dimension and log warning
            min_dim = min(ofer_dim, mica_dim)
            ofer_params = ofer_params[:, :min_dim]
            mica_params = mica_params[:, :min_dim]
            logger.warning(f"FLAME parameter dimension mismatch. Using first {min_dim} dimensions.")
        
        # Weighted fusion
        fused_params = (self.ofer_weight * ofer_params + 
                       self.mica_weight * mica_params)
        
            
        logger.info(f"Fusion completed - output shape: {fused_params.shape}")
        
        return fused_params



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
        # Prepare data
        data = {
            'image': image.cpu().numpy().tolist() if torch.is_tensor(image) else image.tolist(),
            'arcface': arcface.cpu().numpy().tolist() if torch.is_tensor(arcface) else arcface.tolist()
        }
        
        # Send request
        response = requests.post(
            f"{self.api_url}/infer",
            json=data,
            timeout=30  # Adjust timeout as needed
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                return torch.tensor(result['flame_params']).float()
            else:
                raise RuntimeError(f"MICA inference failed: {result['error']}")
        else:
            raise RuntimeError(f"HTTP request failed: {response.status_code}")