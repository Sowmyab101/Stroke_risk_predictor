# Stroke_risk_predictor
Brain Stroke Prediction using ANN Web App 

This project predicts the risk of brain stroke using an Artificial Neural Network (ANN) trained on healthcare data.
It considers lifestyle, medical history, and demographic factors such as BMI, glucose level, hypertension, heart disease, smoking habits, and age to estimate the probability of stroke.

## 🚀 Features
- ANN model built with TensorFlow/Keras  
- Handles class imbalance using weights  
- Streamlit web app for user-friendly prediction  
- Risk levels shown as **Low / Moderate / High**
  
📁 **brain_stroke_risk_predictor**
 ├── train_ann_light.py      # Training script for ANN
 ├── stroke_model_light.keras # Trained ANN model
 ├── scaler.save             # StandardScaler for input preprocessing
 ├── app.py                  # Streamlit app for predictions
 ├── stroke_data.csv         # Dataset (if included)
 ├── README.md               # Project documentation


▶️ **Usage**
1. Train the Model (optional)
python train_ann_light.py

3. Run the Streamlit App
streamlit run app.py
Open in your browser:



🧪 **Model Details**
Architecture: ANN with 2 hidden layers (8 → 4 neurons)
Activation Functions: ReLU, Sigmoid
Optimizer: Adam
Loss: Binary Crossentropy
Handles imbalanced dataset using class weights


**Dataset**
Source: Stroke Prediction Dataset (Kaggle)
Target: stroke (0 = No Stroke, 1 = Stroke)
