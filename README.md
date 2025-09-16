# 🚦 Traffic Flow Forecasting using Spatial CNNs

This project implements a **Spatial Convolutional Neural Network (CNN)** to forecast urban traffic flow.  
It combines **grid-based sensor mapping**, **synthetic/real traffic data**, **deep learning forecasting**, and **unsupervised clustering** for traffic pattern discovery.

---

## ⚡ Features

- Convert **sensor distance matrix** → **2D grid mapping** (via MDS)
- Create **traffic heatmaps sequences**
- Train & tune a **3D CNN** with [Keras Tuner](https://keras.io/keras_tuner/)
- Save **best model** (`models/cnn_model.keras`)
- Visualize **predicted vs actual heatmaps**
- Apply **KMeans clustering** for unsupervised traffic pattern discovery
- Reduce with **PCA** for visualization

---

## 📊 Example Results

### CNN Forecast (Predicted vs Actual Heatmap)

<img src="./results/pred_vs_actual.png" width="600"/>

### Cluster Centers (Traffic Patterns)

<img src="./esults/cluster_centers.png" width="600"/>

### PCA of Traffic Heatmap Clusters

<img src="./results/pca_clusters.png" width="400"/>

---

## 🛠️ Installation

## 1. Clone the repo
git clone https://github.com/hariravi-ds/Traffic-Flow-Forecasting-Using-Spatial-CNNs.git
cd Traffic-Flow-Forecasting-Using-Spatial-CNNs

## 2. Create virtual environment (recommended, macOS/Linux)

python3 -m venv .venv
source .venv/bin/activate

## 3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

## ▶️ Running the Pipeline
python main.py

This will:
Train + tune CNN
Save best model → models/cnn_model.keras
Save results → results/ folder:
pred_vs_actual.png (forecast comparison)
cluster_centers.png (unsupervised clusters)
pca_clusters.png (PCA scatter of clusters)

## 📦 Requirements
See requirements.txt.

Key libraries:
TensorFlow / Keras
Keras-Tuner
scikit-learn
numpy, pandas
matplotlib, seaborn

## 🔮 Next Steps
Replace synthetic generator with real-world datasets (e.g. METR-LA, PeMS)
Add spatio-temporal GNNs for comparison
Experiment with attention-based forecasting