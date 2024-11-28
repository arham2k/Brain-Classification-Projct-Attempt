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

def load_data(image_dir):
    images = []
    labels = []
    yes_count = 0  
    no_count = 0  
    
    for file_name in os.listdir(image_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG')):
            image_path = os.path.join(image_dir, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
            image = cv2.resize(image, (128, 128))  
            images.append(image)
            
            if 'Y' in file_name.upper() or 'YES' in file_name.upper():  
                labels.append(1)  # tumor
                yes_count += 1
            elif 'N' in file_name.upper() or 'NO' in file_name.upper():  
                labels.append(0)  # No tumor
                no_count += 1
    
    print(f"\nTotal 'yes' files (Tumor present): {yes_count}")
    print(f"Total 'no' files (No tumor): {no_count}")
    
    return np.array(images), np.array(labels)

images, labels = load_data(IMAGE_DIR)

images = images / 255.0  
images = images[..., np.newaxis]  

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

def build_cnn(input_shape=(128, 128, 1)):
    model = models.Sequential()
    
    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    
    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification (0 or 1)
    
    return model

cnn_model = build_cnn()
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the CNN model using TensorFlow/Keras
history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=20
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the validation set
val_loss, val_accuracy = cnn_model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Make predictions on the validation set
predictions = cnn_model.predict(X_val)
predictions = (predictions > 0.5).astype(np.uint8)  

# Visualize predictions
# Check if the labels and predictions are correct
# Load the trained model (Ensure this is defined first)
model = tf.keras.models.load_model('brain_tumor_classifier.h5')

# Loop through and display images, ground truth, and predictions
for i in range(5):
    print(f"Image {i} - Ground Truth: {y_val[i]}, Prediction: {predictions[i]}")
    print(f"Raw prediction values: {model.predict(X_val[i:i+1])}")  # Show raw prediction value
    
    plt.figure(figsize=(10, 3))

    # Display the original image
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(X_val[i].squeeze(), cmap='gray')
    plt.axis('on')  

    # Display the Ground Truth label (0 or 1) as a grayscale image
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Label')
    plt.imshow(np.full_like(X_val[i], fill_value=y_val[i]), cmap='gray', vmin=0, vmax=1)
    plt.axis('on')  

    # Display the Predicted label (0 or 1) as a grayscale image
    plt.subplot(1, 3, 3)
    plt.title('Predicted Label')
    plt.imshow(np.full_like(X_val[i], fill_value=(predictions[i] > 0.5)), cmap='gray', vmin=0, vmax=1)
    plt.axis('on')  

    plt.show()


# Save the model
cnn_model.save('brain_tumor_classifier.h5')
# Save predictions
np.save('outputs/predictions.npy', predictions)




