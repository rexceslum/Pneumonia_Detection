from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.optim as optim
import os

# %% Set GPU device
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Data preprocessing and load data
test_dir = './data/chest_xray/test/'

test_dataset = torchvision.datasets.ImageFolder(
        root=test_dir,
        transform=transforms.Compose([
                      transforms.Resize((200,200)),
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
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Using ResNet50

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
# print(model)


# %% Grad-CAM for ResNet50
# Select the target layer for Grad-CAM
target_layer = model.resnet.layer4[-1]  # Last convolutional layer in VGG16

# Initialize Grad-CAM
cam = GradCAM(model=model, target_layers=[target_layer])

# Get predicted outputs from test dataset
test_outputs = model(images.to(device))

# Ensure image_id is within bounds
image_id = 12
if image_id >= len(images):
    raise ValueError(f"image_id {image_id} is out of range. Maximum index is {len(images) - 1}.")

# Use the first image in the batch
input_image = images[image_id].unsqueeze(0).to(device)  # Add batch dimension and send to device
original_image_np = images[image_id].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) for visualization
original_image_np = (original_image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Denormalize
original_image_np = np.clip(original_image_np, 0, 1)

# Generate Grad-CAM heatmap
grayscale_cam = cam(input_tensor=input_image)  # By default, targets=None uses the most confident class

# Overlay Grad-CAM heatmap on the original image
heatmap = show_cam_on_image(original_image_np, grayscale_cam[0], use_rgb=True)

# Normalize the heatmap
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# Get predicted label
_, predicted_classes = torch.max(test_outputs, dim=1)
predicted_classes = predicted_classes.cpu().numpy()
pred_label = list(test_dataset.class_to_idx.keys())[labels[image_id].item()]
predicted_label = list(test_dataset.class_to_idx.keys())[predicted_classes[image_id].item()]

# Display the Grad-CAM result
if predicted_classes[image_id] == labels[image_id]:
    print("Groundtruth for this image:", pred_label)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title(f"{predicted_label}")
    save_path = f"./result/gradcam_{pred_label}_{image_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Grad-CAM visualization saved to {save_path}")
else:
    print(f"Incorrect prediction. Predicted: {predicted_label}, Ground Truth: {pred_label}")


