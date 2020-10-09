from torch import nn
import torch
import numpy as np

class Discriminator(nn.Module):

    def __init__(self,classes,input_acc_size):
        super(Discriminator,self).__init__()

        # label emmbedding
        model = [nn.Embedding(classes,input_acc_size)]
        self.embed_model = nn.Sequential(*model)

        # discrimination
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
        sequence += [nn.Softmax(dim=2)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self,input,labels):
        embeded_label = self.embed_model(labels)
        concatenated = torch.mul(input,embeded_label)
        return self.model(input)

if __name__ == "__main__":
    classes = 10
    batches = 175
    input_size = 180
    model = Discriminator(input_size)
    z = torch.FloatTensor(np.random.normal(0, 1, (batches, 1,input_size)))
    labels = torch.from_numpy(np.random.randint(0,classes-1,batches))

    output = model(z,labels)
    print(output.shape)
    print(output)