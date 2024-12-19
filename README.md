# Brain-Classification-Project-Attempt
 


## About This Project

This is a personal project developed to gain practical experience in using machine learning techniques for medical image classification. As an undergraduate degree holder in Biomedical Engineering (BME) and currently pursuing a master's degree in Electrical and Computer Engineering (ECE), this project serves as an application of machine learning to real-world biomedical challenges. The focus is on classifying brain tumor images using a Convolutional Neural Network (CNN) to identify the presence or absence of a tumor in medical images, particularly MRI scans or CT scans.

This project applies deep learning concepts, including CNNs, to binary image classification tasks and demonstrates the workflow of loading, preprocessing, training, evaluating, and visualizing the predictions made by the model.

## Key Objectives:
- Implement a Convolutional Neural Network (CNN) to classify medical images as tumor present or not.
- Utilize real-world image datasets for model training and validation.
- Visualize and interpret model predictions, including ground truth vs. predicted labels.

## Dataset

The dataset used in this project contains images of brain scans that have been labeled to indicate the presence or absence of a tumor. This dataset was manually curated with the following structure:

- **Yes (Tumor present)**: Images with tumor presence are labeled with 'Y' or 'YES' in the filename.
- **No (No tumor)**: Images without tumor presence are labeled with 'N' or 'NO' in the filename.

For this project, the dataset was collected from publicly available medical image repositories and personal datasets. It is important to note that this dataset is used for educational and research purposes and may not be suitable for clinical use.

### Directory Structure:

/data                  # Directory containing the image dataset (images should be organized here for training)
/outputs               # Folder to store predictions and output files (e.g., trained models, visualizations)
/brain_tumor_classifier.h5  # Trained model saved after completion of training
/requirements.txt      # List of dependencies required to run the project
README.md              # Documentation file providing an overview of the project



The images are in `.jpg`, `.jpeg`, or `.png` formats and are grayscale with a resolution of 128x128 pixels. All images are resized to 128x128 to standardize input sizes for the CNN.

## Requirements

To run this project, you need to install the following Python libraries:

- `numpy`: for numerical operations.
- `opencv-python`: for image processing.
- `scikit-learn`: for model evaluation and data splitting.
- `tensorflow`: for building and training the CNN model.
- `torch`: for potential use with PyTorch (though not actively used in the current version).
- `matplotlib`: for data visualization and plotting.



