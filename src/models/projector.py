import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, expansion=0):
        super(Projector, self).__init__()

        self.linear_1 = nn.Linear(512*expansion, 2048)
        self.linear_2 = nn.Linear(2048, 128)
    
    def forward(self, x):
            
        output = self.linear_1(x)
        output = F.relu(output)

        output = self.linear_2(output)
        
        return output

