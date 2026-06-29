# dataset/
#  ├── train/
#  │    ├── Pneumonia/
#  │    └── Normal/
#  ├── val/
#  │    ├── Pneumonia/
#  │    └── Normal/

# 1. Install Dependencies
# pip install torch torchvision matplotlib scikit-learn
# 2. Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
# 3. Data Preprocessing
# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # convert to 3-channel
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
train_data = datasets.ImageFolder('dataset/train', transform=transform)
val_data = datasets.ImageFolder('dataset/val', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

class_names = train_data.classes

# 4. CNN Model (Transfer Learning: ResNet18)
model = models.resnet18(pretrained=True)

# Modify final layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 6. Training Loop
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(loader)

# 7. Validation Function
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred

# 8. Training Execution
epochs = 10

for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_acc, _, _ = evaluate(model, val_loader)
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# 9. Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

val_acc, y_true, y_pred = evaluate(model, val_loader)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()

# 10. Save Model
torch.save(model.state_dict(), "pneumonia_model.pth")



Data PreprocessingData Preprocessing
