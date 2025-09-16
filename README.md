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
