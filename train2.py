import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import cv2
from torch.utils.checkpoint import checkpoint

class ArtworkDataset(Dataset):
    def __init__(self, video_dir, tablet_dir):
        self.video_dir = video_dir
        self.tablet_dir = tablet_dir
        self.transform = transforms.ToTensor()

        # Get list of video and tablet files
        self.video_files = sorted(os.listdir(video_dir))
        self.tablet_files = sorted(os.listdir(tablet_dir))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # Load last frame of video
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        video = cv2.VideoCapture(video_path)
        success, image = video.read()
        last_frame = None
        while success:
            last_frame = image
            success, image = video.read()
        if last_frame is not None:
            image = Image.fromarray(last_frame)
            image = self.transform(image)

        # Load corresponding tablet
        tablet_path = os.path.join(self.tablet_dir, self.tablet_files[idx])
        tablet = np.load(tablet_path)
        tablet = torch.from_numpy(tablet)


        # Split the tablet into layers along the z-dimension
        tablet_layers = torch.split(tablet, split_size_or_sections=1, dim=2)

        return image, tablet_layers, len(tablet_layers)


class ImageTo3DModel(nn.Module):
    def __init__(self,output_size):
        self.outputsize = output_size
        super(ImageTo3DModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.rnn = nn.GRU(64*8*8, 256*256, batch_first=True)

    def forward(self, x, zlayers):
        # x is a 2D image
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Initialize the hidden state
        h = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)

        # Generate a sequence of 2D layers
        layers = []
        for _ in range(zlayers):  # Replace with the actual number of layers
            x, h = checkpoint(self.rnn, x, h)  # Use checkpointing
            layer = x.view(*self.output_size, 1) # Reshape the output to a 2D layer
            layers.append(layer)

        return layers

# Paths to your image and tablet directories
video_dir = 'autodl-tmp/Videos'
tablet_dir = 'autodl-tmp/tablets'

# Create dataset and data loader
dataset = ArtworkDataset(video_dir, tablet_dir)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Create model, loss function, and optimizer
# Create model, loss function, and optimizer
model = ImageTo3DModel(output_size=(1280, 720))

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.half().to('cuda')

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 2000
model_path = 'Aria.pth'  # specify your path here


for epoch in range(num_epochs):
    for i, (inputs, targets, zlayers) in enumerate(data_loader):
        # Move inputs and targets to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.half().to('cuda')
            targets = [target.half().to('cuda') for target in targets]

        # Forward pass
        outputs = model(inputs, zlayers)

        
        # Compute loss for each layer in the sequence
        loss = 0
        for output, target in zip(outputs, targets):
            loss += criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i)
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()))
    
    # Save the model after each epoch
    torch.save(model.state_dict(), model_path)
