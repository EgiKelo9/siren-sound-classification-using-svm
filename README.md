# Siren Sound Classification Using SVM

This repository contains a project for classifying siren sounds (e.g., ambulance, police, fire truck) using a Support Vector Machine (SVM) classifier. The goal is to distinguish siren sounds from other environmental audio for potential applications such as smart city systems, traffic monitoring, or emergency response support.

## 📁 Project Structure
|── .venv/ # Virtual environment to store dependencies
|── sounds/ # Siren dataset downloaded from Kaggle
|── app.py # Streamlit UI code
|── background.jpg # Background image used in Streamlit
|── module.py # All modules (ex: load audio, preprocessing, SVM model)
|── svm_encoder.pkl # Exported encoder class labels
|── svm_model.pkl # Exported trained SVM model
|── svm_scaler.pkl # Exported standard scaler
|── svm_scratch.ipynb # Jupyter notebook to implement pipeline
|── requirements.txt # Python dependencies
└── README.md # Project overview

## 🚀 How to Use
1. Download dataset from https://www.kaggle.com/datasets/vishnu0399/emergency-vehicle-siren-sounds. It will give you 'sounds' folder.
2. Clone this repository and put the 'sounds' folder inside your cloned project.
3. Create virtual environment and install all dependecies required.
4. You can retrain model in Jupyter notebook and re-export the pickle model, scaler, and encoder.
5. Run Streamlit app and try to classify your siren sounds.
