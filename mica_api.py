# mica_api.py
from flask import Flask, request, jsonify
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append('/Users/amiyachowdhury/Desktop/MICA_test/MICA')

from configs.config import get_cfg_defaults
from utils import util

app = Flask(__name__)

class MICAInference:
    def __init__(self, checkpoint_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = get_cfg_defaults()
        self.cfg.model.testing = True
        
        self.mica = util.find_model_using_name(
            model_dir='micalib.models', 
            model_name=self.cfg.model.name
        )(self.cfg, self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'arcface' in checkpoint:
            self.mica.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            self.mica.flameModel.load_state_dict(checkpoint['flameModel'])
        
        self.mica.eval()
        print(f"MICA API initialized on {self.device}")

mica_model = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.json
        image_data = np.array(data['image'])
        arcface_data = np.array(data['arcface'])
        
        with torch.no_grad():
            images = torch.tensor(image_data).float().to(mica_model.device)
            arcface = torch.tensor(arcface_data).float().to(mica_model.device)
            
            codedict = mica_model.mica.encode(images, arcface)
            opdict = mica_model.mica.decode(codedict)
            
            flame_params = opdict['pred_canonical_shape_vertices'].cpu().numpy().tolist()
            
        return jsonify({
            'success': True,
            'flame_params': flame_params
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    checkpoint_path = './data/pretrained/mica.tar'
    mica_model = MICAInference(checkpoint_path)
    app.run(host='0.0.0.0', port=5010, threaded=False)