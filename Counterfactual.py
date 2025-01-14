import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

# %% Set GPU device
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Data preprocessing and load data
test_dir = './data/chest_xray/test/'

test_dataset = torchvision.datasets.ImageFolder(
    root=test_dir,
    transform=transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
print(test_dataset.class_to_idx)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)
images, labels = next(iter(test_loader))
print(images.shape)
print(labels.shape)

# %% Load the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

        # Replace the fully connected layer according to our problem
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feats, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x

model_save_path = "./model/pytorch_resnet50_model.pth"
model = CNNModel()
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
else:
    model.to(device)
model.eval()

# %% Counterfactual Explanations
# Load one image from the dataset
image, true_label = test_dataset[10]  # Use the first image as an example
image = image.unsqueeze(0).to(device)  # Add batch dimension

# Define target class for counterfactual explanation
target_class = 1 if true_label == 0 else 0  # Flip class for demonstration

# Counterfactual generation parameters
learning_rate = 0.01
num_iterations = 500
regularization_weight = 0.01  # Weight for minimizing perturbation

# Make the image trainable
counterfactual = image.clone().detach().requires_grad_(True)

# Optimization loop
optimizer = torch.optim.Adam([counterfactual], lr=learning_rate)
for iteration in range(num_iterations):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(counterfactual)
    target_score = outputs[0, target_class]
    
    # Regularization: minimize difference from original image
    perturbation = counterfactual - image
    loss = -target_score + regularization_weight * torch.norm(perturbation)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Clip the image values to valid range
    counterfactual.data = torch.clamp(counterfactual.data, 0, 1)
    
    # Check if target achieved
    predicted_class = torch.argmax(F.softmax(outputs, dim=1)).item()
    if predicted_class == target_class:
        print(f"Target class achieved at iteration {iteration}")
        break

# Visualize original and counterfactual images
original_image_np = image[0].permute(1, 2, 0).cpu().detach().numpy()
counterfactual_image_np = counterfactual[0].permute(1, 2, 0).cpu().detach().numpy()

# Denormalize if applicable
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
original_image_np = std * original_image_np + mean
counterfactual_image_np = std * counterfactual_image_np + mean

# Clip values to [0, 1]
original_image_np = np.clip(original_image_np, 0, 1)
counterfactual_image_np = np.clip(counterfactual_image_np, 0, 1)

# Get labels name
predicted_label_name = list(test_dataset.class_to_idx.keys())[target_class]
true_label_name = list(test_dataset.class_to_idx.keys())[true_label]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image_np)
plt.title(f"Original Image\nTrue Label: {true_label_name}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(counterfactual_image_np)
plt.title(f"Counterfactual Image\nTarget Label: {predicted_label_name}")
plt.axis("off")

plt.show()
