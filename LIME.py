from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

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

# %% LIME Explanation
# Function to predict probabilities
def predict_fn(images):
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy()

def slic_segmentation(image):
        return slic(image, n_segments=50, compactness=10)

# Select an image for explanation
image_id = 11
if image_id >= len(images):
    raise ValueError(f"image_id {image_id} is out of range. Maximum index is {len(images) - 1}.")

input_image = images[image_id].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
input_image = (input_image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Denormalize
input_image = np.clip(input_image, 0, 1)

# Get the predicted label
with torch.no_grad():
    outputs = model(images.to(device))
    _, predicted_classes = torch.max(outputs, dim=1)

predicted_label = predicted_classes[image_id].item()
true_label = labels[image_id].item()
predicted_label_name = list(test_dataset.class_to_idx.keys())[predicted_label]
true_label_name = list(test_dataset.class_to_idx.keys())[true_label]
print(f"predicted_label = {predicted_label}, true label = {true_label}")

if predicted_label == true_label:
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Explain the model's prediction on the image
    explanation = explainer.explain_instance(
        input_image,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=2000,  # Number of perturbed samples
        segmentation_fn=slic_segmentation
    )

    # Get the explanation for the predicted class
    lime_result, mask = explanation.get_image_and_mask(
        label=true_label,
        positive_only=True,
        num_features=10,
        hide_rest=True
    )

    # Visualize the original image and LIME explanation
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(input_image, mask))
    plt.axis("off")
    plt.title(f"LIME Explanation - Class: {predicted_label_name}")
    save_path = f"./result/lime_{predicted_label_name}_{image_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
else:
    print(f"Prediction mismatch! Predicted: {predicted_label_name}, Ground Truth: {true_label_name}")
    

