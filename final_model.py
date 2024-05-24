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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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



def reflect_images(images):
    reflected_images = []
    labels = []
    
    # Create the directory if it does not exist
    output_dir = 'dataset_reflected'
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, img in enumerate(images):
        h, w = img.shape[:2]
        
        # Horizontal top mirrored
        top_half = img[:h//2, :]
        mirrored_top_half = cv2.flip(top_half, 0)
        top_mirrored = np.vstack((top_half, mirrored_top_half))
        
        # Horizontal bottom mirrored
        bottom_half = img[h//2:, :]
        mirrored_bottom_half = cv2.flip(bottom_half, 0)
        bottom_mirrored = np.vstack((mirrored_bottom_half, bottom_half))
        
        # Vertical left mirrored
        left_half = img[:, :w//2]
        mirrored_left_half = cv2.flip(left_half, 1)
        left_mirrored = np.hstack((left_half, mirrored_left_half))
        
        # Vertical right mirrored
        right_half = img[:, w//2:]
        mirrored_right_half = cv2.flip(right_half, 1)
        right_mirrored = np.hstack((mirrored_right_half, right_half))
        
        # Save the mirrored images
        top_filename = os.path.join(output_dir, f'image_{idx}_top_mirrored.png')
        bottom_filename = os.path.join(output_dir, f'image_{idx}_bottom_mirrored.png')
        left_filename = os.path.join(output_dir, f'image_{idx}_left_mirrored.png')
        right_filename = os.path.join(output_dir, f'image_{idx}_right_mirrored.png')
        
        cv2.imwrite(top_filename, top_mirrored)
        cv2.imwrite(bottom_filename, bottom_mirrored)
        cv2.imwrite(left_filename, left_mirrored)
        cv2.imwrite(right_filename, right_mirrored)
        
        reflected_images.append(top_mirrored)
        labels.append(0)
        reflected_images.append(bottom_mirrored)
        labels.append(0)
        reflected_images.append(left_mirrored)
        labels.append(1)
        reflected_images.append(right_mirrored)
        labels.append(1)
    
    return reflected_images, labels

# Example usage
reflected_images, labels = reflect_images(images)

# for i in labels[0:10]:
#     print(i)

print(len(reflected_images))
print(len(labels))

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

# Initialize lists to store metrics for each fold
accuracies = []
precision_scores = []
recall_scores = []
f1_scores = []
histories = []
confusion_matrices = []

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f'Fold {fold + 1}')

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

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

    y_true = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_true),
        y=y_true
    )
    class_weights = dict(enumerate(class_weights))

    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_generator = CustomDataGenerator(X_train, y_train, batch_size=16, datagen=datagen)
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=(X_test, y_test),
        class_weight=class_weights
    )

    histories.append(history)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    accuracies.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    confusion_matrices.append(cm)

    print(f'Fold {fold + 1} Classification Report:')
    print(classification_report(y_true_classes, y_pred_classes))

average_accuracy = np.mean(accuracies)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print(f'\nFinal Performance Metrics:')
print(f'Average Accuracy: {average_accuracy * 100:.2f}%')
print(f'Average Precision: {average_precision:.2f}')
print(f'Average Recall: {average_recall:.2f}')
print(f'Average F1-Score: {average_f1:.2f}')

# Plot average training and validation accuracy/loss
average_train_accuracy = np.mean([history.history['accuracy'] for history in histories], axis=0)
average_val_accuracy = np.mean([history.history['val_accuracy'] for history in histories], axis=0)
average_train_loss = np.mean([history.history['loss'] for history in histories], axis=0)
average_val_loss = np.mean([history.history['val_loss'] for history in histories], axis=0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(average_train_accuracy, label='Train Accuracy')
plt.plot(average_val_accuracy, label='Val Accuracy')
plt.title('Average Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(average_train_loss, label='Train Loss')
plt.plot(average_val_loss, label='Val Loss')
plt.title('Average Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot average confusion matrix
average_cm = np.mean(confusion_matrices, axis=0)
plt.figure(figsize=(8, 6))
sns.heatmap(average_cm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
plt.title('Average Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
