import os
import cv2
import numpy as np
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB3
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

# Define directories for training and testing datasets
train_dir = "Brain tumor images/Training"
test_dir = "Brain tumor images/Testing"

# Define labels corresponding to the folder names
labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor", "other"]

# Initialize lists to hold image data and their corresponding labels
X_train = []
y_train = []
image_size = 100  # Standard size to which all images will be resized

# Load training images
for label in labels:
    folderPath = os.path.join(train_dir, label)  # Path to each label's folder
    for image_name in tqdm(os.listdir(folderPath), desc=f"Loading {label} training images"):
        img = cv2.imread(os.path.join(folderPath, image_name))  # Read image
        img = cv2.resize(img, (image_size, image_size))  # Resize image to uniform dimensions
        X_train.append(img)  # Add image to training data
        y_train.append(label)  # Add corresponding label

# Load testing images
for label in labels:
    folderPath = os.path.join(test_dir, label)  # Path to each label's folder
    for image_name in tqdm(os.listdir(folderPath), desc=f"Loading {label} testing images"):
        img = cv2.imread(os.path.join(folderPath, image_name))  # Read image
        img = cv2.resize(img, (image_size, image_size))  # Resize image to uniform dimensions
        X_train.append(img)  # Add image to training data
        y_train.append(label)  # Add corresponding label

# Convert lists to numpy arrays for faster processing
X_train = np.array(X_train)
y_train = np.array(y_train)

# Shuffle data to ensure random distribution of classes
X_train, y_train = shuffle(X_train, y_train, random_state=101)

# Display shapes of data arrays
print("X_train shape: ", X_train.shape)
print("y_train shape:", y_train.shape)

# Split the dataset into training and testing sets (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.85, random_state=23)

# Convert string labels to categorical indices for training data
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)  # Convert to one-hot encoding

# Convert string labels to categorical indices for testing data
y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)  # Convert to one-hot encoding

# Display some samples of processed data
print("X_train sample: ", X_train)
print("y_train sample: ", y_train)

# Load the EfficientNetB3 model pre-trained on ImageNet
# Exclude the top layer to customize for the specific classification task
effnet = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Add custom layers on top of the EfficientNet base model
model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)  # Add global average pooling layer
model = tf.keras.layers.Dropout(rate=0.5)(model)  # Add dropout for regularization
model = tf.keras.layers.Dense(5, activation='softmax')(model)  # Final output layer for 5 classes
model = tf.keras.models.Model(inputs=effnet.input, outputs=model)

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Define callbacks for monitoring and improving the training process
tensorboard = TensorBoard(log_dir='logs')  # For visualizing training logs in TensorBoard
checkpoint = ModelCheckpoint("effnet.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)  # Save best model
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)  # Reduce learning rate on plateau

# Train the model with training data and validation split
history = model.fit(
    X_train, y_train,
    validation_split=0.1,  # Use 10% of training data for validation
    epochs=20,  # Number of training epochs
    verbose=1,  # Show training progress
    batch_size=32,  # Number of samples per batch
    callbacks=[tensorboard, checkpoint, reduce_lr]  # Apply callbacks
)

# Make predictions on the testing data
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)  # Get class with highest probability

# Convert one-hot encoded testing labels back to class indices
y_test_new = np.argmax(y_test, axis=1)

# Print classification report and confusion matrix for evaluation
print(classification_report(y_test_new, pred))
