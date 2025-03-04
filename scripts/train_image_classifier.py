import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def load_preprocessed_data(data_folder):
    # Load preprocessed data
    X_train = np.load(os.path.join(data_folder, "X_train.npy"))
    y_train = np.load(os.path.join(data_folder, "y_train.npy"))
    X_val = np.load(os.path.join(data_folder, "X_val.npy"))
    y_val = np.load(os.path.join(data_folder, "y_val.npy"))

    # Debug: Check data types
    print("y_train dtype:", y_train.dtype)
    print("y_val dtype:", y_val.dtype)
    print("Sample y_train:", y_train[:10])
    print("Sample y_val:", y_val[:10])

    return X_train, X_val, y_train, y_val

def build_model(input_shape=(224, 224, 3)):
    # Load pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history

def save_model(model, model_folder):
    # Create model folder if it doesn't exist
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Save the model
    model.save(os.path.join(model_folder, "amblyopia_detection_model.h5"))
    print(f"Model saved to {model_folder}")

if __name__ == "__main__":
    # Define paths
    data_folder = "data/processed"
    model_folder = "models"

    # Load preprocessed data
    X_train, X_val, y_train, y_val = load_preprocessed_data(data_folder)

    # Build the model
    model = build_model()

    # Train the model
    train_model(model, X_train, y_train, X_val, y_val)

    # Save the model
    save_model(model, model_folder)