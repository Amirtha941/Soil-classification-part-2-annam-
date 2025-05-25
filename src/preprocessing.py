"""
Author: Annam.ai IIT Ropar  
Team Name: AgroMinds AI  
Team Members: Amirtha K, Tharun Babu C  
Leaderboard Rank: 75
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Configuration
IMAGE_SIZE = (128, 128)
DATASET_PATH = "dataset"  # Replace with your actual dataset path
BATCH_SIZE = 32

def preprocessing():
    print("Starting preprocessing...")

    # Data generator with augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Training data generator
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation data generator
    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    print("Preprocessing complete.")
    return train_generator, val_generator
