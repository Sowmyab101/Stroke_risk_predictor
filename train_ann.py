# train_ann_light.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load data
df = pd.read_csv("stroke_data.csv").dropna()

# Encode categorical columns
categorical_cols = ['sex', 'work_type', 'Residence_type', 'smoking_status']
df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)

# Features and target
X = df.drop(columns=['stroke'])
y = df['stroke'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.save")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Class weights to handle imbalance
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {int(cls): weight for cls, weight in zip(np.unique(y_train), weights)}

# Build lighter ANN (faster for inference)
model = Sequential([
    Dense(8, activation='relu', input_shape=(X.shape[1],)),  # fewer neurons
    Dense(4, activation='relu'),                              # smaller hidden layer
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2,
          class_weight=class_weights, verbose=1)

# Save in Keras v3 format
model.save("stroke_model_light.keras")
print("Light ANN model and scaler saved successfully!")
