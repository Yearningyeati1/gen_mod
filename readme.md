# 🧠 A Hybrid Generative-Deterministic Framework for 3D Face Reconstruction

This repository contains the official code for the **B.Tech Project**:

> **“A Hybrid Generative-Deterministic Framework for 3D Face Reconstruction.”**

This project investigates a hybrid framework that fuses the outputs of the metrically accurate **MICA** model (deterministic) and a lightweight, adapted version of **OFER** (generative), called **“Gen-Mod.”**  
The fusion is performed **at inference time**, requiring **no model retraining**.

---

## ⚠️ License and Usage Notice

This project relies on models and code from other research projects.  
Their original licenses **must be respected**:

- **MICA**: Available for *non-commercial research* purposes.  
  → See the [MICA license](https://github.com/Zielon/MICA) for details.

- **OFER**: Available for *non-commercial research* purposes.  
  → See the [OFER project page](https://ofer.is.tue.mpg.de).

- **FLAME (2020)**: Available for *non-commercial research* purposes.  
  → See the [FLAME project page](https://flame.is.tue.mpg.de).

> By using this repository, you agree to adhere to all underlying licenses.  
> This codebase is provided **for non-commercial research purposes only.**

---

## ⚙️ Installation and Setup

This project uses **two Conda environments**:

- `gen_mod` → For running the **fusion pipeline** (this repository)  
- `mica` → For running the **MICA API server**

---

### 🧩 Step 1: Clone This Repository

```bash
git clone https://github.com/Yearningyeati1/gen_mod
cd gen_mod
```

---

### 🧠 Step 2: Setup for “Gen-Mod” (This Repo)

#### Create the Conda Environment

```bash
conda env create -f gen_mod.yml
conda activate gen_mod
```

#### Download OFER Models

1. Go to the [OFER Project Page](https://ofer.is.tue.mpg.de/download.php)  
   (Signup and license agreement required)
2. Download the **IdGen** and **IdRank** network models:
   - `model_idrank.tar`
   - `model_idgen_flame20.tar`
3. Place them into:
   ```
   checkpoint/
   ```

#### Download FLAME Model

1. Go to the [FLAME Project Page](https://flame.is.tue.mpg.de/download.php)  
   (Signup and license agreement required)
2. Download the **FLAME 2020** model.
3. Place the contents into:
   ```
   pretrained/
   ```

---

### 🧮 Step 3: Setup for MICA API Server (Dependency)

The `test_amiya.py` script communicates with the **MICA** model via a **Flask API**.

#### Clone the MICA Repository

```bash
# Navigate outside of this project's folder
git clone https://github.com/Zielon/MICA.git
cd MICA
```

#### Follow MICA’s Setup Instructions

Follow the README in the MICA repository to:
- Create its own Conda environment (e.g., `mica`)
- Download required pretrained models

#### Install Flask and Prepare the API

```bash
# Inside the MICA environment
conda activate mica
pip install flask
```

Copy the `api_server.py` file from this repository into your MICA repository root.

---

## 🚀 How to Run the Demo

You will need **two terminals** open:

---

### 🖥️ Terminal 1: Start the MICA API Server

```bash
# Inside the MICA repository
conda activate mica
python api_server.py
```

You should see a message indicating the Flask server is running, e.g.:
```
 * Running on http://127.0.0.1:5000
```

---

### 🧩 Terminal 2: Run the Fusion Script

```bash
# Inside this project's repository
conda activate gen_mod

python test_amiya.py --cfg './src/configs/config_flameparamdiffusion_flame20.yml'                      --numcheckpoint 3                      --checkpoint1 'checkpoint/model_idrank.tar'                      --checkpoint2 'checkpoint/model_idgen_flame20.tar'                      --imagepath 'data/images/'                      --outputpath 'output'
```

🧾 **Inputs:** Images from the `data/images/` folder  
💾 **Outputs:** Fused 3D mesh results in the `output/` folder

---

## 🧩 Fusion Logic Overview

The **Hybrid Fusion** framework combines deterministic and generative model outputs at the **parameter level** rather than retraining a new network.

- **MICA** provides metrically accurate shape reconstructions (precise geometry).  
- **Gen-Mod (OFER)** captures generative attributes (occlusion robustness).  
- Fusion occurs in the **FLAME parameter space**, ensuring semantic alignment between models.

This hybrid approach retains **MICA’s accuracy** while incorporating **OFER's** complementary information for occlusions from the generative side — without requiring additional training.

---

## 🙏 Acknowledgements

This work builds upon the contributions of the following research projects:

### 🧍‍♂️ MICA: *Towards Metrical Reconstruction of Human Faces*
```bibtex
@inproceedings{zielonka2022mica,
  title={Towards Metrical Reconstruction of Human Faces},
  author={Zielonka, Wojciech and Bolkart, Timo and Thies, Justus},
  booktitle={ECCV},
  year={2022}
}
```

### 😷 OFER: *Occluded Face Expression Reconstruction*
```bibtex
@inproceedings{selvaraju2025ofer,
  title={OFER: Occluded Face Expression Reconstruction},
  author={Selvaraju, Pratheba and Fernandez Abrevaya, Victoria and Bolkart, Timo and Akkerman, Rick and Ding, Tianyu and Amjadi, Faezeh and Zharkov, Ilya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={26985--26995},
  year={2025}
}
```

### 🔥 FLAME: *Learning a Model of Facial Shape and Expression from 4D Scans*
```bibtex
@article{FLAME:2017,
  title = {Learning a model of facial shape and expression from 4D scans},
  author = {Li, Tianye and Bolkart, Timo and Black, Michael J. and Li, Hao and Romero, Javier},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH Asia)},
  volume = {36},
  number = {6},
  pages = {194:1--194:17},
  year = {2017}
}
```

---

## 📸 Project Overview

![Model Architecture](assets/architecture.png)

The pipeline below summarizes the hybrid workflow:

1. Input image(s) → processed by **MICA** and **Gen-Mod** independently.  
2. Outputs fused at **FLAME vertices level** (shape, expression, and pose).  
3. Resulting 3D mesh generated and exported for visualization.

---

## 💡 Contact

For questions, issues, collaborations, or discussions:  
**Amiya Chowdhury**  
National Institute of Technology Rourkela  
📧 [omiya.28.23@gmail.com]

---
