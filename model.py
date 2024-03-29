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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class FineTunedModel(nn.Module):
    def __init__(self, base_model, num_class):
        super(FineTunedModel, self).__init__()
        self.base_model = base_model
        
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, 256)
        self.classifier = nn.Linear(256, num_class)
        
    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x

def ResNet18(num_class):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class)


def ResNet34(num_class):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class)


def ResNet50(num_class):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_class)


def ResNet101(num_class):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_class)


def ResNet152(num_class):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_class)
