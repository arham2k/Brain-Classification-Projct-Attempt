import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn

IMAGE_DIR = '/Users/arhamsheikh/Documents/Coding Projects/ECE1513 Assignments/SelfProject/data'

model = tf.keras.models.load_model('brain_tumor_classifier.h5')

def classify_and_display_images(image_dir):
    for file_name in os.listdir(image_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG')):  
            image_path = os.path.join(image_dir, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
            if image is None:
                print(f"Error: Unable to load image from {image_path}")
                continue
            
            image_resized = cv2.resize(image, (128, 128))
            image_resized = image_resized / 255.0  
            image_resized = image_resized[np.newaxis, ..., np.newaxis]
            
            prediction = model.predict(image_resized)
            prediction = (prediction > 0.5).astype(np.uint8)  
        
            plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap='gray')  
            plt.title(f"Predicted label: {'Tumor' if prediction[0][0] == 1 else 'No Tumor'}")
            plt.axis('on')  
            plt.show()

classify_and_display_images(IMAGE_DIR)