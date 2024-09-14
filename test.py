import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from transformers import ViTModel, ViTConfig
import os
from PIL import Image
# Custom Dataset Class for UTKFace
class UTKFaceDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        age = int(img_path.split('/')[-1].split('_')[0])  # Extract age from filename
        if self.transform:
            image = self.transform(image)
        return image, age

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet stats
])

# Specify the path to the UTKFace dataset
dataset_path = r'D:\UTKFace'
dataset = UTKFaceDataset(dataset_path=dataset_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN + ViT Hybrid Model Definition
class CNN_ViT_Hybrid(nn.Module):
    def __init__(self):
        super(CNN_ViT_Hybrid, self).__init__()

        # Load a pre-trained ResNet model
        self.cnn = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove the last two layers to get feature maps

        # Define ViT configuration
        config = ViTConfig(
            image_size=14,           # Adjust based on the CNN output
            patch_size=1,            # Each feature map patch is 1x1
            num_channels=2048,       # ResNet's feature map channels
            hidden_size=768,         # ViT hidden size
            num_hidden_layers=8,     # Number of transformer blocks
            num_attention_heads=12,  # Number of attention heads
            intermediate_size=3072,  # Feed-forward layer size
        )
        self.vit = ViTModel(config)

        # Classification head for age prediction
        self.fc = nn.Linear(config.hidden_size, 1)  # Output a single value for regression

    def forward(self, x):
        x = self.cnn(x)  # Feature extraction using CNN
        x = x.flatten(2).permute(2, 0, 1)  # Convert feature maps to sequence of patches
        x = self.vit(inputs_embeds=x).last_hidden_state[:, 0, :]  # ViT processing
        x = self.fc(x)  # Age regression output
        return x

# Instantiate and move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_ViT_Hybrid().to(device)

# Define loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for age regression
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)  # Move data to GPU if available

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Print loss after each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
