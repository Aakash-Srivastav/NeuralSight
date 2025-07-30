import tensorflow as tf
# import keras
# pip install tensorflow==2.15.0
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

# 1. Load and Preprocess the Data
def load_and_preprocess_data(dataset_path):
    """
    Loads the image data from the given directory, preprocesses it, and splits it into training and testing sets.

    Args:
        dataset_path (str): Path to the directory containing the image data.  The directory
                        should be structured such that each subdirectory is named after the class,
                        and contains the images for that class.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels, class_names)
               - train_images:  NumPy array of training images, shape (num_train_samples, height, width, channels)
               - train_labels:  NumPy array of training labels, shape (num_train_samples,)
               - test_images:   NumPy array of test images, shape (num_test_samples, height, width, channels)
               - test_labels:   NumPy array of test labels, shape (num_test_samples,)
               - class_names: List of class names (strings).
    """
    # Use image_dataset_from_directory to load data.  This handles directory structure
    # and creates a tf.data.Dataset object.
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',      # Infer labels from subdirectory names
        label_mode='categorical',  # Use one-hot encoding for labels, needed for SMOTE with categorical
        image_size=(64, 64),  # Resize all images to a consistent size
        batch_size=8,          # Batch size for reading images.
        shuffle=True,           # Shuffle the data to ensure randomness
        seed=42                 # Seed for reproducibility
    )

    class_names = full_dataset.class_names  # Extract class names from the dataset.
    num_classes = len(class_names)

    # Pre-allocate space efficiently
    images_np, labels_np = [], []
    for batch in full_dataset:
        images_np.append(batch[0].numpy())
        labels_np.append(batch[1].numpy())

    images_np = np.concatenate(images_np, axis=0)
    labels_np = np.concatenate(labels_np, axis=0)

    # Split the data into training and testing sets (80% train, 20% test).  We do this before
    # applying SMOTE, so that we're generating synthetic samples only for the training set.
    train_images_np, test_images_np, train_labels_np, test_labels_np = train_test_split(
        images_np, labels_np, test_size=0.2, random_state=42, stratify=labels_np
    )

    print(f"Loaded {len(train_images_np)} training images and {len(test_images_np)} testing images.")
    print(f"Classes: {class_names}")
    return train_images_np, train_labels_np, test_images_np, test_labels_np, class_names

# 3. Define the CNN Model
def create_cnn_model(num_classes):
    """
    Creates a CNN model from scratch using TensorFlow/Keras.  The model architecture
    is designed for image classification tasks.

    Args:
        num_classes (int): The number of classes in the dataset.

    Returns:
        tf.keras.Model: A compiled CNN model.
    """
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(64, 64, 3)),
        # Convolutional Block 1: 32 filters, 3x3 kernel, ReLU activation, max pooling
        
        # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 2: 64 filters, 3x3 kernel, ReLU activation, max pooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 3: 128 filters, 3x3 kernel, ReLU activation, max pooling
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the feature maps to a 1D vector
        layers.Flatten(),

        # Fully Connected Layer 1: 512 units, ReLU activation
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Dropout for regularization

        # Output Layer:  num_classes units, softmax activation for multi-class classification
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model.  We use categorical_crossentropy because we have one-hot encoded labels.
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

# 4. Train the Model
def train_model(model, train_images, train_labels, val_images, val_labels, epochs=30):
    """
    Trains the CNN model on the provided training data, with validation data for monitoring performance.

    Args:
        model (tf.keras.Model): The CNN model to train.
        train_images (numpy.ndarray): Array of training images.
        train_labels (numpy.ndarray): Array of training labels.
        val_images (numpy.ndarray): Array of validation images.
        val_labels (numpy.ndarray): Array of validation labels.
        epochs (int, optional): Number of training epochs. Defaults to 30.

    Returns:
        tf.keras.callbacks.History: History object containing training metrics (loss, accuracy, etc.).
    """

    # No data augmentation since the dataset is already augmented
    train_generator = ImageDataGenerator().flow(train_images, train_labels, batch_size=32)

    # No data augmentation for the validation set
    val_generator = ImageDataGenerator().flow(val_images, val_labels, batch_size=32)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=1
    )
    return history

# 5. Evaluate the Model
def evaluate_model(model, test_images, test_labels, class_names):
    """
    Evaluates the trained CNN model on the test data and prints the classification report
    and confusion matrix.  Also calculates and plots the ROC AUC curve.

    Args:
        model (tf.keras.Model): The trained CNN model.
        test_images (numpy.ndarray): Array of test images.
        test_labels (numpy.ndarray): Array of test labels.
        class_names (list): List of class names.
    """
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Make predictions on the test set
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)  # Convert one-hot to class labels
    true_labels = np.argmax(test_labels, axis=1)

    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

# 7. Main Function
def main():
    """
    Main function to orchestrate the loading, preprocessing, augmentation, model creation, training, and evaluation.
    """

    dataset_path = 'data' 

    # Load and preprocess the data
    train_images, train_labels, test_images, test_labels, class_names = load_and_preprocess_data(dataset_path)

    # Create the CNN model
    num_classes = len(class_names)
    model = create_cnn_model(num_classes)

    history = train_model(model, train_images, train_labels, test_images, test_labels, epochs=10) # Increased Epoch

    # Evaluate the model on the test set
    evaluate_model(model, test_images, test_labels, class_names)

    # After training your CNN model
    model.save('backend/models/new_cnn_model.keras')  # Save as keras format

if __name__ == "__main__":
    main()
