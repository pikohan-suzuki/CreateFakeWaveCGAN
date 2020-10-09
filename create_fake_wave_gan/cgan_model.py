# import torch
# from torch import nn
# from create_fake_wave_gan.discriminator import Discriminator
# # from create_fake_wave_gan.generator import Generator
# from gan_loss import GANLoss

# class CGANModel(nn.Module):
#     def __init__(self,lr=0.0002):
#         super(CGANModel).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.cgan = Generator()
        # self.discriminator = Discriminator()

        # self.criterionGAN = GANLoss().to(self.device)

#     def set_input(self,input):
