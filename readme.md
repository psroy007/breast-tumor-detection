# Breast Tumor Detection Using Convolutional Neural Networks (CNN)

This project aims to detect breast tumors in MRI images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained to classify MRI images into two categories: benign and malignant tumors.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Training the Model](#training-the-model)
4. [Testing the Model](#testing-the-model)
5. [Model Architecture](#model-architecture)
6. [Model Training Parameters](#model-training-parameters)
7. [Model Performance](#model-performance)
8. [Future Improvements](#future-improvements)
9. [Installation](#installation)

## Project Structure

- The `train` and `test` folders contain MRI images of benign and malignant tumors, organized in separate subdirectories.
- The model is saved as `saved_model.h5` in the `model` directory.

## Requirements

Install the required dependencies:

                pip install -r requirements.txt

## Training the Model

Ensure you have your dataset organized as follows:
- train/benign – for benign tumor images.
- train/malignant – for malignant tumor images.
- test/benign – for benign tumor test images.
- test/malignant – for malignant tumor test images.

Run the train.py script to train the CNN model.

                python train.py

The model will be trained for 10 epochs using the training data and evaluated on the test data. After training, the model will be saved as saved_model.h5.

## Testing the Model

Once the model is trained and saved, you can use the test.py script to evaluate the model on a test dataset.

                    python test.py

This will load the model, preprocess the test images, and evaluate the accuracy of the model on the test data.

## Model Architecture

The CNN model is constructed as follows:
- Convolutional Layers: Three convolutional layers with 32, 64, and 128 filters, each followed by a MaxPooling layer to reduce spatial dimensions.
- Fully Connected Layer: A fully connected layer with 128 neurons and ReLU activation.
- Output Layer: A dense layer with 1 neuron and sigmoid activation function to output binary classification (benign/malignant).

## Model Training Parameters

- Image Size: The input images are resized to 150x150 pixels.
- Batch Size: The training batch size is set to 32.
- Epochs: The model is trained for 10 epochs.

## Model Performance

The model's performance is evaluated based on accuracy, and it uses binary cross-entropy as the loss function. You can further fine-tune the model or modify the architecture for better performance.

## Future Improvements

- Augment the dataset using techniques like rotation, zooming, and flipping to increase robustness.
- Experiment with deeper or more complex architectures.
- Implement model evaluation using precision, recall, and F1-score for better insights.

## Installation

To set up and run the project locally, follow the steps below:

                git clone https://github.com/psroy007/breast-tumor-detection.git
