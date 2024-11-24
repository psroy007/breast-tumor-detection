import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
test_folder_path = r"D:\breast_tumor_detection\dataset\Breast Cancer Patients MRI's\test"
model_path = r'model/saved_model.h5'

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Image size (same as training)
image_size = (150, 150)

# Get all subfolders (benign/malignant) in the test folder
subfolders = os.listdir(test_folder_path)

# Initialize counters for evaluation
total_images = 0
correct_predictions = 0

# Iterate through each subfolder (class)
for subfolder in subfolders:
    subfolder_path = os.path.join(test_folder_path, subfolder)
    label = 0 if subfolder.lower() == 'benign' else 1  # Assumes folder names are 'benign' and 'malignant'

    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)

        # Process only image files
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename} in {subfolder} folder")

            # Image preprocessing
            image = load_img(file_path, target_size=image_size)  # Load and resize image
            image = img_to_array(image) / 255.0  # Normalize image to [0, 1]
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Predict the class
            prediction = model.predict(image)
            predicted_label = 1 if prediction[0] > 0.5 else 0  # Threshold at 0.5

            # Determine class names
            predicted_class = "Malignant" if predicted_label == 1 else "Benign"
            actual_class = "Malignant" if label == 1 else "Benign"

            # Display prediction
            print(f"Image: {filename}")
            print(f"Predicted: {predicted_class}")
            print(f"Actual: {actual_class}")
            print("-" * 30)

            # Update counters
            total_images += 1
            if predicted_label == label:
                correct_predictions += 1

# Calculate and display accuracy
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"\nTotal images: {total_images}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")