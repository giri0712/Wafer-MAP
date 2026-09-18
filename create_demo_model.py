"""
create_demo_model.py
Creates a quick demo model for testing the Streamlit UI without waiting for full training.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from src.data_loader import IMG_SIZE, DEFECT_CLASSES

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Creating demo CNN model for testing...")

# Build a simple CNN model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(DEFECT_CLASSES), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Save the untrained model
model.save(f"{OUTPUT_DIR}/best_model.keras")
print(f"[INFO] Demo model saved → {OUTPUT_DIR}/best_model.keras")

# Save label classes
np.save(f"{OUTPUT_DIR}/label_classes.npy", DEFECT_CLASSES)
print(f"[INFO] Label classes saved → {OUTPUT_DIR}/label_classes.npy")

print("\n[SUCCESS] Demo model created! You can now test the Streamlit app.")
print("[INFO] The real training will complete in the background...")
