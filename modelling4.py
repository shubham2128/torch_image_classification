import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns



# The key difference in script4 lies in the evaluation approach:

# KFold Cross-Validation: Script4 employs K-Fold cross-validation for model evaluation.
# It splits the data into num_folds (e.g., 5) sets. In each fold:
# It iterates through each fold using kf.split.
# It splits the data into training and testing sets for that specific fold.
# It trains a new model using the training set with data augmentation and class weights.
# It evaluates the model's performance on the corresponding testing set.
# This process is repeated for all folds.
# This approach provides a more robust evaluation of the model's generalization ability compared 
# to a single train-test split. By training and evaluating on different data subsets, K-Fold helps 
# reduce the chance of overfitting to a specific training set.

# Here's a breakdown of the key changes:

# KFold Initialization: Defines a KFold object with the desired number of folds (num_folds) and sets 
# shuffle=True to randomly shuffle the data before splitting.
# Fold Loop: Iterates through each fold using
# for fold, (train_index, test_index) in enumerate(kf.split(X, y)).
# Data Splitting: Within the loop, splits the data into training and testing sets based on the current
# fold's indices.
# Model Training and Evaluation: Trains a new model for each fold using the training data with data
# augmentation and class weights. Then, evaluates the model on the corresponding testing set and 
# prints the test accuracy.
# By evaluating the model on multiple non-overlapping folds, K-Fold cross-validation provides a 
# more comprehensive assessment of the model's performance on unseen data.



# Load and process images
folder_path = '/Users/shubham/Documents/vinove/dataset'

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    print(f'length of images 0 : {len(images[0])}')
    print(f'length of images 0 : {len(images[1])}')
    print(f'length of images 2 : {len(images[2])}')
    return images


images = load_images(folder_path)
# Generate reflected images
def reflect_images(images):
    reflected_images = []
    labels = []
    
    for img in images:
        # Horizontal reflection (vertical flip)
        h_reflect = cv2.flip(img, 0)
        reflected_images.append(h_reflect)
        labels.append(0)  # Horizontal reflection
        
        # Vertical reflection (horizontal flip)
        v_reflect = cv2.flip(img, 1)
        reflected_images.append(v_reflect)
        labels.append(1)  # Vertical reflection
    
    return reflected_images, labels

reflected_images, labels = reflect_images(images)

# Preprocess images
def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (128, 128))  # Resize images to 128x128
        img = img.astype('float32') / 255.0  # Normalize pixel values
        processed_images.append(img)
    return np.array(processed_images)

X = preprocess_images(reflected_images)
y = to_categorical(labels, num_classes=2)

# Initialize KFold with specified number of folds
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define custom data generator
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, datagen):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.datagen = datagen

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_X = self.X[start_index:end_index]
        batch_y = self.y[start_index:end_index]
        return batch_X, batch_y

# Iterate over each fold
# Initialize lists to store metrics for each fold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Initialize lists to store metrics for each fold
accuracies = []
precision_scores = []
recall_scores = []
f1_scores = []
histories = []
confusion_matrices = []

# Iterate over each fold
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f'Fold {fold + 1}')

    # Split the data into training and testing sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Calculate class weights
    y_true = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_true),
        y=y_true
    )
    class_weights = dict(enumerate(class_weights))

    # # Build the model
    # model = Sequential([
    #     Input(shape=(128, 128, 3)),  
    #     Conv2D(32, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     Dense(2, activation='softmax')
    # ])

    model = Sequential([
            Input(shape=(128, 128, 3)),  # Input layer with image dimensions
            Conv2D(16, (3, 3), activation='relu'),  # Reduce filter count in first layer (e.g., 16 instead of 32)
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),  # Reduce filter count in second layer (e.g., 32 instead of 64)
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),  # Reduce filter count in third layer (e.g., 64 instead of 128)
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),  # Reduce number of units in dense layer (e.g., 64 instead of 128)
            Dropout(0.3),  # Adjust dropout rate if needed
            Dense(2, activation='softmax')  # Output layer for 2 classes
            ])
    
    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with data augmentation and class weights
    train_generator = CustomDataGenerator(X_train, y_train, batch_size=32, datagen=datagen)
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=(X_test, y_test),
        class_weight=class_weights
    )

    # Store history
    histories.append(history)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    # Append metrics to lists
    accuracies.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    confusion_matrices.append(cm)

    # Print classification report for the current fold
    print(f'Fold {fold + 1} Classification Report:')
    print(classification_report(y_true_classes, y_pred_classes))

# Calculate and print average metrics
average_accuracy = np.mean(accuracies)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print(f'\nFinal Performance Metrics:')
print(f'Average Accuracy: {average_accuracy * 100:.2f}%')
print(f'Average Precision: {average_precision:.2f}')
print(f'Average Recall: {average_recall:.2f}')
print(f'Average F1-Score: {average_f1:.2f}')

# Plot training and validation accuracy/loss for each fold
for i, history in enumerate(histories):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Fold {i + 1} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Fold {i + 1} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Plot confusion matrices for each fold
for i, cm in enumerate(confusion_matrices):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Fold {i + 1} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plot average confusion matrix
average_cm = np.mean(confusion_matrices, axis=0)
plt.figure(figsize=(8, 6))
sns.heatmap(average_cm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
plt.title('Average Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
