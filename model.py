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
                kernel_size=3,                      
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=3,                   
                padding=1,
                ),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),                
        )
        
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,              
                out_channels=64,            
                kernel_size=3,                   
                padding=1,
                ),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),                
        )
        # calculate the last conv2 output size
        with torch.no_grad():
            dummy = torch.zeros((1, *input_size))
            x = self.conv3(self.conv2(self.conv1(dummy)))
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
        # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(fc_size, 500)
        self.fc2 = nn.Linear(500,num_classes)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = torch.flatten(x, 1) 
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        return output 