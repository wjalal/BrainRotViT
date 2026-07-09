import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class ConvBlock(nn.Module):
    """A convolutional block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.silu = nn.SiLU()
        
        # 1x1 convolution for channel matching in residual connection (if needed)
        self.needs_projection = (in_channels != out_channels)
        if self.needs_projection:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv(x)
        out = self.silu(out)
        out = self.pool(out)
        
        # Residual connection - project identity if needed
        if self.needs_projection:
            # Match dimensions using 1x1 convolution
            identity = self.projection(identity)
            
            # Handle spatial dimension differences - make sure the identity dimensions match the output
            if identity.shape[2:] != out.shape[2:]:
                identity = nn.functional.interpolate(
                    identity, 
                    size=out.shape[2:],
                    mode='nearest'
                )
        else:
            # Handle only spatial dimension differences if needed
            if identity.shape[2:] != out.shape[2:]:
                identity = nn.functional.interpolate(
                    identity, 
                    size=out.shape[2:],
                    mode='nearest'
                )
        
        # Add residual connection where possible - only when dimensions match
        if identity.shape == out.shape:
            out = out + identity
        
        return out

class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape):
        super(AgePredictionCNN, self).__init__()

        print("AgePredictionCNN Shape:", input_shape)
        
        # Define convolutional blocks with residual connections
        self.conv_block1 = ConvBlock(1, 8, kernel_size=(10, 60), stride=1)
        self.conv_block2 = ConvBlock(8, 4, kernel_size=(5, 15), stride=1)
        self.conv_block3 = ConvBlock(4, 1, kernel_size=(2, 6), stride=1)

        self.flatten = nn.Flatten()

        # Fully connected layers (fc1 dimensions are calculated dynamically)
        self.fc1 = None  # Placeholder to be initialized dynamically
        self.fc1_bn = None  # Placeholder for batch normalization after fc1
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 30% probability
        self.fc3 = nn.Linear(129, 1)  # Adding 1 for the `Sex` input

        self.silu = nn.SiLU()
        self.initialize_fc1(input_shape)

    def initialize_fc1(self, input_shape):
        # Create a sample input to pass through the convolutional layers
        sample_input = torch.zeros(1, *input_shape)
        x = self.conv_block1(sample_input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        flattened_size = x.numel()
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc1_bn = nn.LayerNorm(512)

    def forward(self, x, sex):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)

        if self.fc1 is None:
            raise ValueError("fc1 layer has not been initialized. Call `initialize_fc1` with the input shape.")

        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.silu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.silu(x)
        x = self.dropout(x)

        # Concatenate `Sex` input
        x = torch.cat((x, sex.unsqueeze(1)), dim=1)
        x = self.fc3(x)

        return x