import torch.utils.data as data
import torch

class Dataset(data.Dataset):
    def __init__(self, chest_acc,chest_gyro,pocket_acc,pocket_gyro):
        self.chest_acc  = torch.Tensor(chest_acc).double()
        self.chest_gyro = torch.Tensor(chest_gyro).double()
        self.pocket_acc  = torch.Tensor(pocket_acc).double()
        self.pocket_gyro = torch.Tensor(pocket_gyro).double()

    def __len__(self):
        return len(self.chest_acc)

    def __getitem__(self, index):
        return  self.pull_item(index)
        
    def pull_item(self, index):
        return self.chest_acc[index],self.chest_gyro[index],self.pocket_acc[index],self.pocket_gyro[index]