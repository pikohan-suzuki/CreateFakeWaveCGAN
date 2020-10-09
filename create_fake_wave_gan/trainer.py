import torch
from torch import nn
try:
    from create_fake_wave_gan.discriminator import Discriminator
    from create_fake_wave_gan.generator import Generator
    from create_fake_wave_gan.dataset import Dataset
    from create_fake_wave_gan.dataloader import DataLoader
except Exception as e:
    from discriminator import Discriminator
    from generator import Generator
    from dataset import Dataset
    from balanced_batch_sampler import BalancedBatchSampler
from gan_loss import GANLoss
import glob
import json
import numpy as np
import time

class Trainer():
    def __init__(self,acc_input_size,classes,lr=0.0002):
        super(Trainer,self).__init__()
        self.acc_input_size = acc_input_size
        self.classes = classes
        self.class_dict = dict()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cgan = Generator(classes=classes,input_acc_size=acc_input_size).double().to(self.device)
        self.discriminator = Discriminator(classes=classes,input_acc_size=acc_input_size).double().to(self.device)

        self.criterionGAN = GANLoss().to(self.device)
        self.optimizer_G = torch.optim.Adam(self.cgan.parameters(),lr=lr)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),lr=lr)


    def train(self,epochs,batch_size,acc_input_size):
        dataloader = self.create_data_loader(batch_size)
        print("use '{}'".format(self.device))
        for epoch in range(epochs):
            start_time = time.time()
            s_time = start_time
            for i,data in enumerate(dataloader):
                acc_data,labels = data[0][0].to(self.device),data[1][0].to(self.device)

                z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, 1,acc_input_size))).to(self.device)
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                self.optimizer_G.zero_grad()
                fake_labels = torch.from_numpy(np.random.randint(0,classes,batch_size)).to(self.device)
                fake_wave = self.cgan(z,fake_labels).to(self.device)
                g_loss = self.criterionGAN(self.discriminator(fake_wave,fake_labels),True)
                g_loss.backward()

                for param in self.discriminator.parameters():
                    param.requires_grad = True
                fake_wave = self.cgan(z,labels)
                self.optimizer_D.zero_grad()
                d_loss_real = self.criterionGAN(self.discriminator(acc_data,labels),True)
                d_loss_fake = self.criterionGAN(self.discriminator(fake_wave,labels),False)
                d_loss = 0.5 * torch.add(d_loss_real,d_loss_fake)
                d_loss.backward()


                self.optimizer_D.step()
                self.optimizer_G.step()


            end_time = time.time()
            print(f"{epoch+1}/{epochs}  {int((end_time-start_time)*1000)}ms",end="")
            print("  g_loss: {:1.4}  d_loss {:1.4}".format(g_loss,d_loss))
            start_time = end_time
    # print(z.shape)

    def get_class_annotation(self,ano_str:str):
        if ano_str not in self.class_dict:
            self.class_dict[ano_str] = len(self.class_dict)
        return self.class_dict[ano_str]

    def create_data_loader(self,batch_size=1,is_train=True):
        type_key = "train" if is_train else "test"
        data_files = glob.glob("./dataset/{}/*".format(type_key))
        chest_acc_list = []
        label_list = []
        for file in data_files:
            with open(file) as f:
                loaded = json.load(f)
                chest_acc_list += [loaded['chest_acc']]
                label_list += [self.get_class_annotation(loaded['label'])]
        num_inputs = len(chest_acc_list)
        dataset = Dataset(torch.Tensor(chest_acc_list).view(num_inputs,1,self.acc_input_size),torch.Tensor(label_list))
        sampler = BalancedBatchSampler(dataset, self.classes, batch_size//self.classes)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,sampler = sampler,num_workers=2)
        return dataloader
        
if __name__ == "__main__":
    epochs = 10000
    batch_size = 200
    acc_input_size = 64
    classes = 5

    trainer = Trainer(acc_input_size,classes)
    trainer.train(epochs=epochs,batch_size=batch_size,acc_input_size=acc_input_size)
    