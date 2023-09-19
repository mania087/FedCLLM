import torch
import torch.nn.functional as F
from torch import nn

class CnnModel(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(CnnModel, self).__init__()

        assert len(input_size) == 3, "size should be a tuple of (channel, height, width)"

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=input_size[0],              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=5,              
                stride=1,                   
                padding=2,
                ),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),                
        )
        # calculate the last conv2 output size
        with torch.no_grad():
            dummy = torch.zeros((1, *input_size))
            x = self.conv2(self.conv1(dummy))
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
        # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(fc_size, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = torch.flatten(x, 1)      
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output 