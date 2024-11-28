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

# Load the data (image + mask for segmentation)
def load_data(image_dir):
    images = []
    masks = []
    labels = []
    yes_count = 0  
    no_count = 0  
    
    for file_name in os.listdir(image_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG')):
            image_path = os.path.join(image_dir, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale for simplicity
            image = cv2.resize(image, (128, 128))  # Resize to match model input size
            images.append(image)
            
            # Generate mask based on filename or other criteria
            # Here, the mask is a binary mask (1 where tumor is, 0 otherwise)
            mask = np.zeros_like(image)
            if 'Y' in file_name.upper() or 'YES' in file_name.upper():  
                labels.append(1)  # Tumor present
                yes_count += 1
                mask[image > 100] = 1  # A simple thresholding for example
            elif 'N' in file_name.upper() or 'NO' in file_name.upper():  
                labels.append(0)  # No tumor
                no_count += 1
            masks.append(mask)
    
    print(f"\nTotal 'yes' files (Tumor present): {yes_count}")
    print(f"Total 'no' files (No tumor): {no_count}")
    
    return np.array(images), np.array(masks), np.array(labels)

# Load images and their corresponding masks
images, masks, labels = load_data(IMAGE_DIR)

# Normalize images to [0, 1] and expand dimensions for CNN
images = images / 255.0
images = images[..., np.newaxis]  # Add channel dimension

masks = masks / 255.0  # Normalize masks
masks = masks[..., np.newaxis]  # Add channel dimension

# Split dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Define a segmentation model
def build_segmentation_model(input_shape=(128, 128, 1)):
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
    
    # Output Layer for Segmentation (same size as input image)
    model.add(layers.Dense(128 * 128, activation='sigmoid'))
    model.add(layers.Reshape((128, 128, 1)))  # Reshape to match input size
    
    return model

# Build and compile the model
segmentation_model = build_segmentation_model()
segmentation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = segmentation_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=20
)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the validation set
val_loss, val_accuracy = segmentation_model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Make predictions on the validation set
predictions = segmentation_model.predict(X_val)
predictions = (predictions > 0.5).astype(np.uint8)  # Convert to binary mask

def highlight_tumor_in_image(image, mask):
    # Convert grayscale to BGR (ensure image is uint8)
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit (0-255 range)
    image_with_mask = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
    
    # Highlight tumor in green
    image_with_mask[mask == 1] = [0, 255, 0]  # Set tumor pixels to green
    return image_with_mask


# Visualize predictions and tumor highlight
for i in range(5):
    print(f"Image {i} - Ground Truth: {y_val[i].squeeze()}, Prediction: {predictions[i].squeeze()}")
    
    plt.figure(figsize=(10, 3))

    # Display the original image
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(X_val[i].squeeze(), cmap='gray')
    plt.axis('on')

    # Display the Ground Truth label (0 or 1) as a grayscale image
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(y_val[i].squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.axis('on')

    # Display the Predicted mask
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(predictions[i].squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.axis('on')

    plt.show()

    # Highlight the tumor in the image using the predicted mask
    highlighted_image = highlight_tumor_in_image(X_val[i].squeeze(), predictions[i].squeeze())
    
    plt.figure(figsize=(10, 3))
    plt.imshow(highlighted_image)
    plt.title(f"Highlighted Tumor (Prediction: {predictions[i].squeeze()})")
    plt.axis('off')
    plt.show()

# Save the trained model
segmentation_model.save('brain_tumor_segmentation_model.h5')

# Save predictions
np.save('outputs/predictions.npy', predictions)
