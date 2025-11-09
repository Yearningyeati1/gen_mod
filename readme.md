A Hybrid Generative-Deterministic Framework for 3D Face Reconstruction
This repository contains the official code for the B.Tech project, "A Hybrid Generative-Deterministic Framework for 3D Face Reconstruction."

This project investigates a hybrid framework that fuses the outputs of the metrically-accurate MICA model (deterministic) and a lightweight, adapted version of OFER (generative), which we call "Gen-Mod." The fusion is performed at inference time and does not require model retraining.

⚠️ License and Usage Notice
This project relies on models and code from other research projects. Their original licenses must be respected:

MICA: The MICA model and code are available for non-commercial research purposes. Please see the MICA license for details.

OFER: The OFER model is available for non-commercial research purposes.

FLAME: The FLAME 2020 model is available for non-commercial research purposes.

By using this repository, you agree to adhere to all underlying licenses. This codebase is also provided for non-commercial research purposes only.

⚙️ Installation and Setup
This project requires two separate Conda environments:

gen_mod (This Repo): For running the main fusion script.

mica (Dependency): For running the MICA API server.

Step 1: Clone This Repository
Bash

git clone [https://github.com/Yearningyeati1/gen_mod]
cd [gen_mod]
Step 2: Setup for "Gen-Mod" (This Repo)
Create the Conda Environment:

Bash

conda env create -f gen_mod.yml
conda activate gen_mod
Download OFER Models:

Go to the OFER Project Page.[https://ofer.is.tue.mpg.de/download.php](signup,license agreement required)

Download the IdGen and IdRank network models.

Place the downloaded model files (e.g., model_idrank.tar, model_idgen_flame20.tar) into the checkpoint/ folder in this repository.

Download FLAME Model:

Go to the FLAME Project Page.[https://flame.is.tue.mpg.de/download.php](signup,license agreement required)

Register and download the FLAME 2020 model.

Place the FLAME2020 folder (or its contents as required by your code) into the pretrained/ folder.

Step 3: Setup for MICA API Server (Dependency)
The test_amiya.py script communicates with the MICA model via a Flask API. You must set this up in a separate location.

Clone the MICA Repository (in a new folder):

Bash

# Navigate *outside* of this project's folder
git clone https://github.com/Zielon/MICA.git
cd MICA
Follow MICA's Setup Instructions:

Follow the README.md in the MICA repository to set up their environment and download their required models. This will likely involve creating a new Conda environment (e.g., conda activate mica).

Install Flask and Prepare the API:

Bash

# While in the MICA environment (e.g., 'conda activate mica')
pip install flask
Copy the api_server.py file from this repository into the root of the MICA repository you just cloned.

🚀 How to Run the Demo
You will need two terminals open.

Terminal 1: Start the MICA API Server
Navigate to your MICA repository folder.

Activate the MICA conda environment.

Run the API server:

Bash

# Inside the MICA repository
conda activate mica 
python api_server.py
You should see a message indicating the Flask server is running (e.g., on http://127.0.0.1:5000).

Terminal 2: Run the Fusion Script
Navigate to this project's repository folder.

Activate the gen_mod conda environment.

Run the test script:

Bash

# Inside this project's repository
conda activate gen_mod

python test_amiya.py --cfg './src/configs/config_flameparamdiffusion_flame20.yml' \
                     --numcheckpoint 3 \
                     --checkpoint1 'checkpoint/model_idrank.tar' \
                     --checkpoint2 'checkpoint/model_idgen_flame20.tar' \
                     --imagepath 'data/images/' \
                     --outputpath 'output'
This script will process images from the data/images/ folder.

The final fused 3D mesh outputs will be saved in the output/ folder.

🙏 Acknowledgements
This work builds heavily on the contributions of the following projects. We are grateful to the original authors for making their code and models available.

MICA: Towards Metrical Reconstruction of Human Faces

@inproceedings{zielonka2022mica,
    title={Towards Metrical Reconstruction of Human Faces},
    author={Zielonka, Wojciech and Bolkart, Timo and Thies, Justus},
    booktitle={ECCV},
    year={2022}
}
OFER: OFER: Occluded Face Expression Reconstruction

@inproceedings{selvaraju2025ofer,
    title={OFER: Occluded Face Expression Reconstruction},
    author={Selvaraju, Pratheba and Fernandez Abrevaya, Victoria and Bolkart, Timo and Akkerman, Rick and Ding, Tianyu and Amjadi, Faezeh and Zharkov, Ilya},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={26985--26995},
    year={2025}
}
FLAME: Learning a model of facial shape and expression from 4D scans

@article{FLAME:2017,
    title = {Learning a model of facial shape and expression from 4D scans},
    author = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    volume = {36},
    number = {6},
    year = {2017},
    pages = {194:1--194:17}
}
