
# Remaining Useful Life Prediction for Turbofan Engines  
### Using CNN and LSTM Models  
**Team: Cheng-Hung Chung & Jiaxuan Yan**

---

##  Project Overview
We predict Remaining Useful Life (RUL) of turbofan engines, using the NASA **CMAPSS** dataset (FD001–FD004).  
We implement two neural network architectures:

- **1D Convolutional Neural Network (CNN)**
- **Long Short-Term Memory (LSTM)**
- **CNN+LSTM** 

All models share:

- The same **24-D input features** (3 operating settings + 21 sensors)
- The same **sliding-window preprocessing** (window length `w = 35, 50`)
- The same **training hyperparameters** (epochs, batch size, LR, loss)

Our result show that in this task, LSTM average validation loss (MAE/RMSE/NASA) all perform better than CNN, which show that LSTM can learn better in **degrading trend tasks**.

---

## 📂 **Repository Structure**
| File/Folder | Description |
|---|---|
| `cnn.ipynb` | CNN training & evaluation |
| `lstm.ipynb_final` | LSTM baseline training & evaluation |
| `lstm+cnn_final.ipynb`| LSTM baseline + CNN-LSTM hybrid |
| `README.md` | Documentation file (this file) |
| `Team_Model_Card.docx` | Model card document |
| `Test_Result` | contain training and validation curves |


---

##  **How to Run**

### 1. Environment settings

- Python 3.10+
- PyTorch 2.x + CUDA (if available)
- Ubuntu / macOS / Google Colab /...

### 2. Install dependencies
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### 3. Prepare Dataset
Download `CMAPSSData.zip` to the repository root.  
The notebooks will automatically unzip and extract data into `CMAPSSData/`.

### 4. Train Models
Open the notebook → **Run All** → Training begins automatically,

→Validation performance is printed for every epoch.

---

##  **Data format and layout expectations**

### DATA Input Feature (24‑D)
| Feature Type | Description |
|---|---|
| Operating conditions | `op1`, `op2`, `op3` |
| Sensor values | `s1` … `s21` |

project_root/
└─ CMAPSSData/
   ├─ train_FD001.txt
   ├─ train_FD002.txt
   ├─ train_FD003.txt
   ├─ train_FD004.txt
   ├─ test_FD001.txt
   ├─ test_FD002.txt
   ├─ test_FD003.txt
   ├─ test_FD004.txt
   ├─ RUL_FD001.txt
   ├─ RUL_FD002.txt
   ├─ RUL_FD003.txt
   └─ RUL_FD004.txt

### Sliding Window
- Window lengths evaluated: **35** and **50**
- Each sample uses the **latest W cycles** per engine

### Normalization Strategy
| Dataset | Operating Conditions | Normalization |
|---|:---:|---|
| FD001, FD003 | Single | Global mean/std |
| FD002, FD004 | Multiple | K‑Means condition‑wise normalization |

> Matches the instructor’s dataset considerations.

---

##  Models

| Model | Architecture Summary |
|---|---|
| CNN | 3×Conv1D → Flatten → 2×FC → RUL |
| LSTM | 2‑layer LSTM → ReLU‑FC → RUL |
| CNN-LSTM | Conv1D(24→32, k=3) → 1-layer LSTM(64 hidden) → FC(64→32→1, ReLU) → RUL |

### Training hyperparameters

**Common settings**

- Epochs: **20**
- Batch size: **64**
- Optimizer: Adam
- Loss: **MSE** (primary training loss)
- RUL clipping: **125**

**Per-model learning rate**

- **CNN baseline**: learning rate = **1e-3**
- **LSTM baseline**: learning rate = **1e-3**
- **CNN-LSTM hybrid**: learning rate = **1e-4**

---

##  Validation Results  

| Dataset   | Window | **CNN RMSE** | **LSTMRMSE** |                                          
| --------- | :----: | :----------: | :-----------: |
| **FD001** |   35   |   15.300544  |   12.829396   | w=35 operating condition              
| **FD001** |   50   |   13.438353  |   12.719482   | w=50 offer more context, improves predictions       
| **FD002** |   35   |   17.136366  |   15.869294   | FD002 & 004 performed not as good as 001 &003
| **FD002** |   50   |   15.390746  |   14.086301   | 
| **FD003** |   35   |   14.890630  |   13.271042   | 
| **FD003** |   50   |   13.019991  |   11.646756   | 
| **FD004** |   35   |   15.726580  |   13.344357   | 
| **FD004** |   50   |   14.368486  |   13.117378   | 


 Add:
- Loss learning curves
- RMSE/NASA score plots

---

##  Key Insights
- LSTM better captures degradation progression
- FD002 & FD004 significantly harder → multi‑condition challenges
- CNN tends to overfit under shifting operating regimes (especially FD004)

---

##  Known Limitations
- No post‑training smoothing / uncertainty estimation
- Validation only (test RUL requires decoding final cycles)
- Sliding window fixed length for all engines

---

##  Reproducibility Support
- All hyperparameters fixed and logged
- Random seed set to ensure consistent runs
- Training procedures fully contained in notebooks
- Model checkpoint saving included

---

##  Team Contribution
| Member | Contribution |
|---|---|
| Cheng‑Hung Chung | LSTM pipeline + LSTM+CNN pipeline+condition‑aware preprocessing + training evaluation|
| Jiaxuan Yan | CNN model + training evaluation + visualization |

---

## 📎 References
- NASA CMAPSS Turbofan Engine Simulator Dataset
- EE541 Final Project Guidelines

---
