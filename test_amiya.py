# inspired by OFER code which was inspired by MICA code

import os
import sys
import torch
import re
import numpy as np
import random
from pytorch_lightning import seed_everything

from src.tester_amiya import Tester2
from src.simplified_tester import SimpleTester
from src.testerrank import Tester as TesterRank

from src.models.baselinemodels.flameparamdiffusion_model import FlameParamDiffusionModel
from src.models.baselinemodels.flameparamrank_model import FlameParamRankModel


def random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

if __name__ == '__main__':


    from src.configs.config import get_cfg_defaults, update_cfg
    from src.configs.config import parse_args
    deviceid = torch.device("cpu")
    #torch.cuda.empty_cache()
    #num_gpus = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print("num gpus = ", num_gpus)
    #print("device_id", deviceid, flush=True)
    print("device = ", device, flush=True)

    cfg_file = './src/configs/config_flameparamdiffusion_flame20.yml' 
    cfg_rank = get_cfg_defaults()
    if cfg_file is not None:
        cfg_rank = update_cfg(cfg_rank, cfg_file)



    cfg, args = parse_args()
####################################################################
# Regular Inference
    # print(f"Input image path: {args.imagepath}")
    # print(f"Output path: {args.outputpath}")


    # cfg_rank.train.resume_checkpoint = args.checkpoint1 
    # cfg_rank.model.sampling = 'ddim'
    # cfg_rank.net.arch = 'archv4'
    # cfg_rank.varsched.num_steps = 1000
    # cfg_rank.varsched.beta_1 = 1e-4
    # cfg_rank.varsched.beta_T = 1e-2
    # cfg_rank.train.resume=True
    # cfg_rank.train.resumepretrain = False
    # cfg_rank.model.expencoder = 'arcfarl'
    # cfg_rank.model.preexpencoder = 'arcface'
    # cfg_rank.model.prenettype = 'preattn'
    # cfg_rank.model.numsamples = 10
    # cfg_rank.model.usenewfaceindex = True
    # cfg_rank.model.istrial = False
    # cfg_rank.net.losstype = 'Softmaxlistnetloss'
    # cfg_rank.net.numattn = 1
    # cfg_rank.net.predims = [300,50,10]
    # cfg_rank.model.flametype = 'flame20'
    # cfg_rank.dataset.flametype = 'flame20'
    # cfg_rank.model.nettype = 'listnet'
    # cfg_rank.net.rankarch = 'scorecb1listnet'
    # cfg_rank.net.shape_dim = 5355
    # cfg_rank.net.context_dim = 1024
    # cfg_rank.model.testing = True
    # seed_everything(1)
    # model_rank = FlameParamRankModel(cfg_rank, 'cpu')
    # #print(model_rank)

    # testerrank = TesterRank(model_rank, cfg_rank, deviceid)
    # testerrank.model.load_model()


    # cfg1 = cfg.clone()
    # cfg2 = cfg.clone()

    # cfg1.model.sampling = 'ddim'
    # cfg1.model.with_exp = False
    # cfg1.model.expencoder = 'arcface'
    # cfg1.net.flame_dim = 300
    # cfg1.net.arch = 'archv4'
    # cfg1.net.context_dim = 512
    # cfg1.model.nettype = 'preattn'
    # cfg1.net.dims = [300,50,10]
    # cfg1.net.numattn = 1
    # cfg1.train.resume = True
    # cfg1.dataset.flametype = 'flame20'
    # cfg1.model.flametype = 'flame20'
    # cfg1.train.resume_checkpoint = args.checkpoint2
    # cfg1.model.testing = True
    # model1 = FlameParamDiffusionModel(cfg1, 'cpu')
    # #print(model1)
    # model1.eval()

    # # --- MODIFICATION 2: Instantiate SimpleTester ---
    # tester = SimpleTester([model1], cfg, [cfg1], deviceid, args, rankmodel=testerrank, mica_api_url='http://localhost:5010')
    
    # # --- MODIFICATION 3: Call process_folder() ---
    # tester.process_folder()

###########################################################################
# NOW EVALUATION
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
    seed_everything(1)
    model_rank = FlameParamRankModel(cfg_rank, 'cpu')
    #print(model_rank)

    testerrank = TesterRank(model_rank, cfg_rank, deviceid)
    testerrank.model.load_model()


    cfg1 = cfg.clone()
    cfg2 = cfg.clone()

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
    cfg1.train.resume_checkpoint = args.checkpoint2
    cfg1.model.testing = True
    model1 = FlameParamDiffusionModel(cfg1, 'cpu')
    #print(model1)
    model1.eval()

    tester = Tester2([model1], cfg, [cfg1], deviceid, args,rankmodel=testerrank, mica_api_url='http://localhost:5010')
    # Add MICA configuration
    # mica_api_url = 'http://localhost:5010'  # or None to disable MICA


    #tester = Tester2([model1], cfg, [cfg1], deviceid, args,rankmodel=testerrank, mica_api_url=args.mica_url)
    
    tester.test_now()
################################################################################
