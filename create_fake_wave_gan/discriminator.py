from torch import nn
import torch
import numpy as np

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        sequence = []
        sequence += [nn.Conv1d(1,8,kernel_size = 4,stride=2,padding=1),
        nn.LeakyReLU(0.2,True)]
        sequence += [nn.Conv1d(8,16,kernel_size = 4,stride=2,padding=1),
        nn.BatchNorm1d(16),
        nn.LeakyReLU(0.2,True)]
        sequence += [nn.Conv1d(16,32,kernel_size = 4,stride=2,padding=1),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(0.2,True)]
        sequence += [nn.Conv1d(32,32,kernel_size = 4,stride=2,padding=1),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(0.2,True)]
        sequence += [nn.Conv1d(32,1,kernel_size = 4,stride=2,padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self,input):
        return self.model(input)

if __name__ == "__main__":
    model = Discriminator()
    input_data = torch.Tensor(np.random.rand(4,1,64))
    output = model(input_data)
    print(output.shape)