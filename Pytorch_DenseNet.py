import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# %% Set GPU device
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Data preprocessing and load data
train_dir = './data/chest_xray/train/'
test_dir = './data/chest_xray/test/'
val_dir = './data/chest_xray/val/'

train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=transforms.Compose([
                      transforms.Resize((200,200)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(20),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
)
test_dataset = torchvision.datasets.ImageFolder(
        root=test_dir,
        transform=transforms.Compose([
                      transforms.Resize((200,200)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
)
val_dataset = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=transforms.Compose([
                      transforms.Resize((200,200)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
)
print(train_dataset.class_to_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=True
)
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)


# %% Building the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)  # Using DenseNet121

        # Replace the classifier layer according to our problem
        in_feats = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_feats, 2)

    def forward(self, x):
        x = self.densenet(x)
        return x

model_save_path = "./model/pytorch_densenet121_model.pth"
model = CNNModel()
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
else:
    model.to(device)
    
print(model)


# %% Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 20
training_losses = []
validation_losses = []
training_accuracy = []
validation_accuracy = []

# Iterate x epochs over the train data
if not os.path.exists(model_save_path):
    for i in range(epochs):
        model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Training pass
            # Sets the gradient to zero
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Labels are automatically one-hot-encoded
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # This is where the model learns by backpropagating and accumulates the loss for mini batch
            loss.backward()
            # Then optimizes its weights here
            optimizer.step()
            # Calculate training accuracy
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        # Average loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        train_accuracy = 100 * correct_train / total_train
        training_accuracy.append(train_accuracy)
        print(f"Epoch {i+1}/{epochs} - Training Loss: {avg_train_loss}, Training Accuracy: {train_accuracy}")
        torch.cuda.empty_cache()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                # Calculate validation accuracy
                _, val_preds = torch.max(val_outputs, 1)
                correct_val += (val_preds == val_labels).sum().item()
                total_val += val_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        val_accuracy = 100 * correct_val / total_val
        validation_accuracy.append(val_accuracy)
        print(f"Epoch {i+1}/{epochs} - Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")
        torch.cuda.empty_cache()
    
    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
else:
    print(f"Use saved model exists at {model_save_path}")


# %% Test
model.eval()  # Set model to evaluation mode
test_loss = 0.0
correct_test = 0
total_test = 0
all_preds = []
all_labels = []
correct_samples = []
incorrect_samples = []

with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_outputs = model(test_inputs)
        test_loss += criterion(test_outputs, test_labels).item()
        # Calculate testing accuracy
        _, test_preds = torch.max(test_outputs, 1)
        correct_test += (test_preds == test_labels).sum().item()
        total_test += test_labels.size(0)
        # Store predictions and true labels for classification report
        all_preds.extend(test_preds.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())
        # Collect correctly and incorrectly classified samples
        for i in range(len(test_labels)):
            if test_preds[i] == test_labels[i]:
                correct_samples.append((test_inputs[i].cpu(), test_labels[i].cpu()))
            else:
                incorrect_samples.append((test_inputs[i].cpu(), test_labels[i].cpu()))

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct_test / total_test

# Print testing results
print(f"Testing Loss: {avg_test_loss}, Testing Accuracy: {test_accuracy}")
print(f"Number of Images Tested: {total_test}")


# %% Evaluation metrics
# Loss evolution plot
if training_losses:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', marker='o', color='green', linestyle='-')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss', marker='o', color='red', linestyle='-')
    plt.title('Loss Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_path_loss = f"./data/result/densenet121_loss_evolution.png"
    plt.savefig(save_path_loss, dpi=300, bbox_inches='tight')

# Accuracy evolution plot
if training_accuracy:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_accuracy) + 1), training_accuracy, label='Training Accuracy', marker='o', color='green', linestyle='-')
    plt.plot(range(1, len(validation_accuracy) + 1), validation_accuracy, label='Validation Accuracy', marker='o', color='red', linestyle='-')
    plt.title('Accuracy Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    save_path_accuracy = f"./data/result/densenet121_accuracy_evolution.png"
    plt.savefig(save_path_accuracy, dpi=300, bbox_inches='tight')

# Classification report
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

# Confusion matrix plot
cm = confusion_matrix(all_labels, all_preds)
cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linecolor = 'black', linewidth = 1, xticklabels=train_dataset.classes, 
            yticklabels=train_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
save_path_confusion_matrix = f"./data/result/densenet121_confusion_matrix.png"
plt.savefig(save_path_confusion_matrix, dpi=300, bbox_inches='tight')

# Display some correctly classified samples
print("\nCorrectly Classified Samples:")
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, (img, label) in enumerate(correct_samples[:5]):
    img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Denormalize
    img = torch.clamp(img, 0, 1)  # Clamp values between 0 and 1
    axes[i].imshow(img.numpy())
    axes[i].set_title(f"True: {train_dataset.classes[label]}")
    axes[i].axis("off")
save_path_correctly_classified = f"./data/result/densenet121_correctly_classified.png"
plt.savefig(save_path_correctly_classified, dpi=300, bbox_inches='tight')

# Display some incorrectly classified samples
print("\nIncorrectly Classified Samples:")
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, (img, label) in enumerate(incorrect_samples[:5]):
    img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Denormalize
    img = torch.clamp(img, 0, 1)  # Clamp values between 0 and 1
    axes[i].imshow(img.numpy())
    axes[i].set_title(f"True: {train_dataset.classes[label]}")
    axes[i].axis("off")
save_path_incorrectly_classified = f"./data/result/densenet121_incorrectly_classified.png"
plt.savefig(save_path_incorrectly_classified, dpi=300, bbox_inches='tight')
