import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import KFold, train_test_split 
from sklearn.metrics import classification_report , confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        
        reflected_images.extend([top_mirrored, bottom_mirrored, left_mirrored, right_mirrored])
        labels.extend([0, 0, 1, 1])
    
    return reflected_images, labels

reflected_images, labels = reflect_images(images)

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(reflected_images, labels, transform=transform)

# Split dataset into training/validation and test sets
train_val_images, test_images, train_val_labels, test_labels = train_test_split(reflected_images, labels, test_size=0.2, stratify=labels, random_state=42)

# Create datasets
train_val_dataset = CustomDataset(train_val_images, train_val_labels, transform=transform)
test_dataset = CustomDataset(test_images, test_labels, transform=transform)

# Stratified K-Fold Cross Validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

all_train_losses = []
all_val_losses = []
all_train_accs = []
all_val_accs = []
all_true_labels = []
all_predicted_labels = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_images, train_val_labels)):
    print(f'Fold {fold + 1}')
    
    train_sub = Subset(train_val_dataset, train_idx)
    val_sub = Subset(train_val_dataset, val_idx)
    
    train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=64, shuffle=False)
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # Validation
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
        
        val_loss = running_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        
        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}')

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accs.append(train_accs)
    all_val_accs.append(val_accs)

# Print classification report for validation set
print('\nValidation Set Classification Report:')
print(classification_report(all_true_labels, all_predicted_labels))
print(confusion_matrix(all_true_labels, all_predicted_labels))


# Evaluate on the final test set
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model.eval()
true_test_labels = []
predicted_test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_test_labels.extend(labels.cpu().numpy())
        predicted_test_labels.extend(predicted.cpu().numpy())

print('\nTest Set Classification Report:')
print(classification_report(true_test_labels, predicted_test_labels))
print(confusion_matrix(true_test_labels, predicted_test_labels))
# Plotting
plt.figure(figsize=(12, 4))

# Average train and validation losses across folds
avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(avg_train_losses, label='Train Loss')
plt.plot(avg_val_losses, label='Val Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Average train and validation accuracies across folds
avg_train_accs = np.mean(all_train_accs, axis=0)
avg_val_accs = np.mean(all_val_accs, axis=0)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(avg_train_accs, label='Train Accuracy')
plt.plot(avg_val_accs, label='Val Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
