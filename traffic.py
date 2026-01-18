import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Global variables
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    """
    images = []
    labels = []

    print(f"Loading data from {data_dir}...")

    # Loop through every category folder
    for category in range(NUM_CATEGORIES):
        folder_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(folder_path):
            continue

        # Progress indicator
        if category % 5 == 0:
            print(f"Reading category {category}...")

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Read image
            img = cv2.imread(file_path)

            if img is not None:
                # Resize to standard dimensions
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(category)

    print(f"Successfully loaded {len(images)} images.")
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = tf.keras.models.Sequential([

        # 1. First Convolutional Layer: Detects low-level features (edges)
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # 2. Second Convolutional Layer: Detects mid-level features (shapes)
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # 3. Third Convolutional Layer: Detects high-level features (complex patterns)
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

        # Flatten into a 1D vector
        tf.keras.layers.Flatten(),

        # 4. Dense (Hidden) Layer for classification logic
        tf.keras.layers.Dense(256, activation="relu"),

        # 5. Dropout to prevent the AI from "memorizing" specific images (overfitting)
        tf.keras.layers.Dropout(0.5),

        # 6. Output Layer: Probability for each of the 43 sign types
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile using Adam optimizer and cross-entropy loss
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
