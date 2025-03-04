import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(image_folder, labels_csv, img_size=(224, 224)):
    # Load labels (no header in the CSV file)
    labels_df = pd.read_csv(labels_csv, header=None, names=["image_filename", "label"])
    
    # Initialize lists to store images and labels
    images = []
    labels = []

    # Loop through each row in the CSV
    for _, row in labels_df.iterrows():
        image_filename = row['image_filename']
        label = row['label']

        # Handle the special case where label is '[Negative,Positive]'
        if label == '[Negative,Positive]':
            label = '[Positive]'  # Convert to '[Positive]'

        # Clean and convert the label to numeric
        if label == '[Positive]':
            label = 1  # Amblyopia
        elif label == '[Negative]':
            label = 0  # No Amblyopia
        else:
            raise ValueError(f"Unknown label: {label}")

        # Load and preprocess the image
        image_path = os.path.join(image_folder, image_filename)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, img_size)  # Resize image
            image = image / 255.0  # Normalize to [0, 1]

            # Add to lists
            images.append(image)
            labels.append(label)

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)  # Ensure labels are integers

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

def save_preprocessed_data(output_folder, X_train, X_val, y_train, y_val):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save preprocessed data
    np.save(os.path.join(output_folder, "X_train.npy"), X_train)
    np.save(os.path.join(output_folder, "X_val.npy"), X_val)
    np.save(os.path.join(output_folder, "y_train.npy"), y_train)
    np.save(os.path.join(output_folder, "y_val.npy"), y_val)
    print(f"Preprocessed data saved to {output_folder}")

if __name__ == "__main__":
    # Define paths
    image_folder = "data/test"
    labels_csv = "data/testlabels.csv"
    output_folder = "data/processed"

    # Load and preprocess images
    X_train, X_val, y_train, y_val = load_and_preprocess_images(image_folder, labels_csv)

    # Save preprocessed data
    save_preprocessed_data(output_folder, X_train, X_val, y_train, y_val)