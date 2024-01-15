import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Assuming you have a 3D numpy array 'data' and a 2D grayscale image 'image'
# data.shape = (width, height, time)
# image.shape = (width, height)

data = np.load('/root/autodl-tmp/tablets/nk480tablet.npy')
video_path = '/root/autodl-tmp/videos/NIGHT KOI DEMO 480.mov'
video = cv2.VideoCapture(video_path)
success, image = video.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(data.shape,image.shape)
'''
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.encoder(input.view(1, 1, -1), hidden)
        output = self.decoder(output, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.tconv = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.tconv(x)
        return x

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''
'''
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, output_size)

    def forward(self, input, hidden):
        # Flatten the input
        input = input.view(-1)
        output, hidden = self.encoder(input.view(1, 1, -1), hidden)
        output = self.decoder(output, hidden)
        # Reshape the output back into 2D
        output = output.view(input_size, -1)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

# Adjust the input and output sizes to match the dimensions of the picture
input_size = data.shape[0] * data.shape[1]
output_size = data.shape[0] * data.shape[1]'''
class Seq2Seq(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(Seq2Seq, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.Gates = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                               out_channels=4 * self.hidden_channels,  # for input, forget, cell, and output gates
                               kernel_size=self.kernel_size,
                               padding=self.padding)

    def forward(self, input_tensor, cur_state):
        if cur_state is None:
            h_cur, c_cur = self.init_hidden(input_tensor)
        else:
            h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        gates = self.Gates(combined)  # compute state of input, forget, cell, and output gates
        i_gate, f_gate, c_gate, o_gate = torch.split(gates, self.hidden_channels, dim=1)  # split the gates
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)

        c_next = f_gate * c_cur + i_gate * c_gate  # next cell state
        h_next = o_gate * torch.tanh(c_next)  # next hidden state

        return h_next, (h_next, c_next)


    def init_hidden(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()
        h_init = torch.zeros(batch_size, self.hidden_channels, height, width).to(input_tensor.device)
        c_init = torch.zeros(batch_size, self.hidden_channels, height, width).to(input_tensor.device)
        return h_init, c_init

# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of input channels, in this case it's 1 because the image is grayscale
input_channels = 1
# Number of output channels, you can adjust this value according to your needs
hidden_channels = 64
# Kernel size of the ConvLSTM, you can adjust this value according to your needs
kernel_size = (3, 3)

model = Seq2Seq(input_channels, hidden_channels, kernel_size)
model = model.to(device)  # Move model to GPU

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training
for epoch in range(10):  # number of epochs
    for z in range(data.shape[2] - 1):
        #print(epoch,z)
        optimizer.zero_grad()
        input_tensor = torch.tensor(data[:, :, z], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)  # Move tensors to GPU
        target_tensor = torch.tensor(data[:, :, z + 1], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)  # Move tensors to GPU
        loss = 0

        h_cur, c_cur = model.init_hidden(input_tensor)
        output, (h_next, c_next) = model(input_tensor, (h_cur, c_cur))
        loss = criterion(output, target_tensor)

        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    # Save the model after each epoch
    if epoch%10==0:
        torch.save(model.state_dict(),'Aria.pth')

import cv2
import numpy as np

# Assuming 'model' is your trained ConvLSTM model
# and 'input_image' is your initial input image

# Move the model to evaluation mode
model.eval()

# Load the initial input image
input_image = cv2.imread('a.png', cv2.IMREAD_GRAYSCALE)
input_image = torch.from_numpy(input_image).float().unsqueeze(0).unsqueeze(0).to(device)

# Initialize the hidden state
h_cur, c_cur = model.init_hidden(input_image)

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('output.mp4', fourcc, 30, (input_image.shape[2], input_image.shape[3]))

# Generate 300 layers
for i in range(300):
    output, (h_cur, c_cur) = model(input_image, (h_cur, c_cur))

    # Convert the output tensor to a numpy array
    output_array = output.detach().cpu().numpy()

    # Normalize the output array to the range [0, 255]
    output_array = ((output_array - output_array.min()) * (255 / (output_array.max() - output_array.min()))).astype(np.uint8)

    # Add an extra dimension to the output array and repeat it along that dimension to create a 3-channel image
    output_array = np.repeat(output_array[..., np.newaxis], 3, axis=-1)

    # Write the output array to the video file
    video.write(output_array)

    # Use the output as the input for the next layer
    input_tensor = output

# Release the VideoWriter
video.release()
