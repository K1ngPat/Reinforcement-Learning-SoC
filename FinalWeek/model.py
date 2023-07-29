# imports
import torch.nn as nn
import torch.nn.functional as F

# model
class ConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, convolution_filters):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.convolution_filters = convolution_filters

        # convolutional layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[1], out_channels=self.convolution_filters, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(num_features=self.convolution_filters),
            nn.ReLU()
        )

        # residual layer
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=self.convolution_filters, out_channels=self.convolution_filters, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=self.convolution_filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.convolution_filters, out_channels=self.convolution_filters, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(self.convolution_filters)
        )

        # residual layers
        self.residual_layers = nn.ModuleList(
            [self.residual for _ in range(self.num_hidden_layers)]
        )

        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=self.convolution_filters, out_channels=2, kernel_size=(1, 1), padding=0, stride=(1, 1)),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2*self.input_dim[2]*self.input_dim[3], out_features=self.output_dim[1]),
            nn.Sigmoid() # probability = (0, 1)
        )

        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=self.convolution_filters, out_channels=1, kernel_size=(1, 1), padding=0, stride=(1, 1)),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1*self.input_dim[2]*self.input_dim[3], out_features=self.convolution_filters),
            nn.ReLU(),
            nn.Linear(in_features=self.convolution_filters, out_features=1),
            nn.Tanh() # value = (-1, 1)
        )

        
    def forward(self, x):
        # convolutional layer
        x = self.conv(x)

        # residual layers
        for residual_layer in self.residual_layers:
            x_res = residual_layer(x)
            x += x_res
            x = F.relu(x)

        # policy output
        policy_output = self.policy_head(x) # shape: [-1, 73 * 8 * 8]

        # value output
        value_output = self.value_head(x) # shape: [-1, 1]

        return policy_output, value_output