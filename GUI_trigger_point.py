import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic


# global variables
model_save_path = "./model/pytorch_resnet50_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = {0: 'NORMAL', 1: 'PNEUMONIA'}
gradcam_save_path = "./result/gradcam_result.png"
lime_save_path = "./result/lime_result.png"

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


def predict(input_image):
    # Transform the input image
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension and send to the appropriate device
    
    # Load ResNet50 model
    model = CNNModel()
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
    else:
        model.to(device)
    model.eval()
    
    # Make the prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)  # Get the index of the highest score
        
    # Map the predicted index to the class label
    predicted_class = classes[predicted_idx.item()]
    
    # Apply XAI explanations
    apply_gradcam(model, input_tensor)
    apply_lime(model, input_tensor, predicted_idx.item())
    
    return predicted_class


def apply_gradcam(model, input_image):
    # Select the target layer for Grad-CAM
    target_layer = model.resnet.layer4[-1]  # Last convolutional layer in VGG16

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Prepare input image numpy
    input_image_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) for visualization
    input_image_np = (input_image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Denormalize
    input_image_np = np.clip(input_image_np, 0, 1)

    # Generate Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=input_image)  # By default, targets=None uses the most confident class

    # Overlay Grad-CAM heatmap on the original image
    heatmap = show_cam_on_image(input_image_np, grayscale_cam[0], use_rgb=True)

    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Save Grad-CAM result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image_np)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title(f"Grad-CAM explanation")
    plt.savefig(gradcam_save_path, dpi=300, bbox_inches='tight')


def apply_lime(model, input_image, predicted_label):
    def predict_fn(images):
        images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    def slic_segmentation(image):
            return slic(image, n_segments=50, compactness=10)
        
    # Prepare input image numpy
    input_image_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()   # Convert to (H, W, C) for visualization
    input_image_np = (input_image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Denormalize
    input_image_np = np.clip(input_image_np, 0, 1)
    
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Explain the model's prediction on the image
    explanation = explainer.explain_instance(
        input_image_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=2000,  # Number of perturbed samples
        segmentation_fn=slic_segmentation
    )

    # Get the explanation for the predicted class
    lime_result, mask = explanation.get_image_and_mask(
        label=predicted_label,
        positive_only=True,
        num_features=10,
        hide_rest=True
    )

    # Visualize the original image and LIME explanation
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image_np)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(input_image_np, mask))
    plt.axis("off")
    plt.title(f"LIME explanation")
    plt.savefig(lime_save_path, dpi=300, bbox_inches='tight')


def get_performance():
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

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True
    )
    
    # Load ResNet50 model
    model = CNNModel()
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
    else:
        model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    # Evaluate testing dataset
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)
            _, test_preds = torch.max(test_outputs, 1)
            # Store predictions and true labels
            y_pred.extend(test_preds.cpu().numpy())
            y_true.extend(test_labels.cpu().numpy())
        
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    return accuracy, precision, recall, f1








from PIL import Image

# Load an image (replace 'path_to_image.jpg' with the actual path)
image_path = "C:/Users/dylum/MasterCode/Pneumonia_Detection/data/chest_xray/test/PNEUMONIA/person15_virus_46.jpeg"
input_image = Image.open(image_path)

# Make a prediction
predicted_class = predict(input_image)
print(f"Predicted Label: {predicted_class}")