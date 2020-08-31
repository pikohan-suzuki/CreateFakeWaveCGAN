import torch
from torch import nn, optim
import numpy as np

class ResnetGenerator(nn.Module):
    def __init__(self,input_nc=1,output_nc=1,ngf=64,n_blocks=4):
        super(ResnetGenerator,self).__init__()
        model = [nn.ReflectionPad1d(1),
                 nn.Conv1d(input_nc,ngf,kernel_size=8,padding=0,bias=True),
                 nn.InstanceNorm1d(ngf),
                 nn.ReLU(True)]

        n_downsampling=2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv1d(ngf*mult,ngf*mult*2,kernel_size = 3,stride=2,padding=1,bias=True),
                      nn.InstanceNorm1d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf*mult)]
        
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling -i)
            model += [nn.ConvTranspose1d(ngf * mult,int(ngf * mult /2),
                      kernel_size=4,stride = 2,padding=1,output_padding=1,bias=True),
                      nn.InstanceNorm1d(int(ngf * mult/2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad1d(3)]
        model += [nn.Conv1d(ngf,output_nc,kernel_size = 8,padding=1)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self,input_acc,input_gyro,labels):
        return self.model(input)


class Generator(nn.Module):

    def __init__(self,classes,input_acc_size,gyro_channel):
        super(Generator,self).__init__()
        
        # label emmbedding
        model = [nn.Embedding(classes,input_acc_size)]
        self.embed_model = nn.Sequential(*model)

        # acc 
        model = []



    def forward(self,input_acc,input_gyro,labels):
        return self.embed_model(labels)


class ResnetBlock(nn.Module):

    def __init__(self,dim):
        super(ResnetBlock,self).__init__()
        self.conv_block = self.build_conv_block(dim)
    
    def build_conv_block(self,dim):
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad1d(1)]
        conv_block += [nn.Conv1d(dim,dim,kernel_size = 3,padding=0,bias=True),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)]
        conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad1d(1)]
        conv_block += [nn.Conv1d(dim,dim,kernel_size=3,padding=0,bias=True),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)]
        return nn.Sequential(*conv_block)

    def forward(self,x):
        out = x + self.conv_block(x)
        return out

if __name__ == "__main__":
    gyro_channel = 3
    classes = 10
    batches = 10
    gen = Generator(classes=classes,input_acc_size=180,gyro_channel=gyro_channel)
    # input_acc_wave = torch.Tensor(np.random.rand(batches,1,180))
    # input_gyro_wave = torch.Tensor(np.random.rand(batches,gyro_channel,180))
    z = torch.normal(mean=0.5,std=0.2,size =(batches,))
    labels = np.random.randint(0,classes-1,batches)
    one_hot_labels = torch.from_numpy(np.identity(classes,dtype=np.int64)[labels])

    output_wave = gen(input_acc_wave,input_gyro_wave,one_hot_labels)
    print(output_wave.shape)
