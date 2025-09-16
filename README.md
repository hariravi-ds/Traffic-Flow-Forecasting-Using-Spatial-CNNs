🚦 Traffic Flow Forecasting with CNNs

📌 Project Overview
This project explores traffic flow prediction using Convolutional Neural Networks (CNNs) and unsupervised clustering. We simulate traffic speeds from sensor data, transform them into spatial heatmaps, and train a 3D CNN to forecast traffic patterns.
We also compare results with unsupervised clustering (KMeans + PCA) for anomaly detection and pattern discovery.

🛠️ Key Features
Data Simulation – Generates synthetic traffic speeds from a distance.csv sensor network file.
Sensor Mapping – Uses Multidimensional Scaling (MDS) to project sensors into a 2D grid.
Heatmap Generation – Creates spatio-temporal heatmaps of traffic speed.
Deep Learning Forecasting – Trains a 3D CNN with KerasTuner for hyperparameter optimization.
Unsupervised Learning – Applies KMeans clustering with PCA visualization for pattern discovery.
Evaluation – Compares CNN predictions vs. ground truth with MAE/MSE metrics.
Visualization – Interactive plots of predicted vs. actual traffic heatmaps.

🚀 Getting Started
1. Clone the repository
git clone https://github.com/hariravi-ds/Traffic-Flow-Forecasting-Using-Spatial-CNNs
cd traffic-flow-forecasting

2. Install dependencies
pip install -r requirements.txt

3. Run the notebook
jupyter notebook notebooks/traffic_forecasting.ipynb

📊 Results
CNN Forecasting
MAE ≈ 4.5 on synthetic traffic data
Predictions capture spatial traffic patterns reasonably well

KMeans Clustering
Revealed distinct traffic states (e.g., “free flow” vs. “congestion”)
PCA visualization helps interpret traffic dynamics

🔮 Future Work
Use real-world datasets (e.g., PEMS-BAY, METR-LA).
Try Graph Neural Networks (GNNs) for spatio-temporal modeling.
Deploy as a web app (Flask/Dash) for live predictions.
Add more evaluation metrics: RMSE, MAPE.

📖 References
Yu et al. (2017). Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.
Li et al. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.

# 🚦 Traffic Flow Forecasting using Spatial CNNs

This project implements a **Spatial Convolutional Neural Network (CNN)** to forecast urban traffic flow.  
It combines **grid-based sensor mapping**, **synthetic/real traffic data**, **deep learning forecasting**, and **unsupervised clustering** for traffic pattern discovery.

---

## 📂 Project Structure

Traffic-Flow-Forecasting-Using-Spatial-CNNs/
│── data/ # input distance.csv (sensor network)
│── models/ # trained CNN models (.keras)
│── notebooks/ # exploratory notebook
│── results/ # generated plots & heatmaps
│── src/ # source code modules
│ ├── cleaning.py # data cleaning utilities
│ ├── data_prep.py # sensor mapping & sequence creation
│ ├── model.py # CNN builder for Keras Tuner
│ ├── clustering.py # KMeans & PCA clustering
│ ├── visualization.py # plotting helpers
│ ├── utils.py # save/load utils
│ └── init.py
│── main.py # entry point for full pipeline
│── requirements.txt
│── README.md


---


## ⚡ Features

- Convert **sensor distance matrix** → **2D grid mapping** (via MDS)
- Generate **synthetic traffic data** (or plug in real-world datasets)
- Create **traffic heatmaps sequences**
- Train & tune a **3D CNN** with [Keras Tuner](https://keras.io/keras_tuner/)
- Save **best model** (`models/cnn_model.keras`)
- Visualize **predicted vs actual heatmaps**
- Apply **KMeans clustering** for unsupervised traffic pattern discovery
- Reduce with **PCA** for visualization

---

## 📊 Example Results

### CNN Forecast (Predicted vs Actual Heatmap)

<img src="results/pred_vs_actual.png" width="600"/>

### Cluster Centers (Traffic Patterns)

<img src="results/cluster_centers.png" width="600"/>

### PCA of Traffic Heatmap Clusters

<img src="results/pca_clusters.png" width="400"/>

---

## 🛠️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/Traffic-Flow-Forecasting-Using-Spatial-CNNs.git
cd Traffic-Flow-Forecasting-Using-Spatial-CNNs

### 2. Create virtual environment (recommended, macOS/Linux)

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependencies

bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt


### ▶️ Running the Pipeline

bash
Copy code
python main.py
This will:

Train + tune CNN

Save best model → models/cnn_model.keras

Save results → results/ folder:

pred_vs_actual.png (forecast comparison)

cluster_centers.png (unsupervised clusters)

pca_clusters.png (PCA scatter of clusters)

### 📦 Requirements

See requirements.txt.
Key libraries:

TensorFlow / Keras

Keras-Tuner

scikit-learn

numpy, pandas

matplotlib, seaborn

### 🔮 Next Steps

Replace synthetic generator with real-world datasets (e.g. METR-LA, PeMS)

Add spatio-temporal GNNs for comparison

Experiment with attention-based forecasting