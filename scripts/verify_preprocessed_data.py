import os
import numpy as np
import matplotlib.pyplot as plt

def verify_preprocessed_data(data_folder):
    # Load preprocessed data
    X_train = np.load(os.path.join(data_folder, "X_train.npy"))
    y_train = np.load(os.path.join(data_folder, "y_train.npy"))
    X_val = np.load(os.path.join(data_folder, "X_val.npy"))
    y_val = np.load(os.path.join(data_folder, "y_val.npy"))

    # Print dataset statistics
    print(f"Training images: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Validation images: {X_val.shape}")
    print(f"Validation labels: {y_val.shape}")

    # Display a few sample images from the training set
    plt.figure(figsize=(10, 10))
    for i in range(9):  # Display 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[i])
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the path to the preprocessed data
    data_folder = "data/processed"

    # Verify the preprocessed data
    verify_preprocessed_data(data_folder)