import torch.utils.data as data
import torch

class Dataset(data.Dataset):
    def __init__(self, chest_data,label):
        self.chest_data  = torch.Tensor(chest_data).double()
        self.label  = torch.Tensor(label).long()

    def __len__(self):
        return len(self.chest_data)

    def __getitem__(self, index):
        return  self.pull_item(index)
        
    def pull_item(self, index):
        return self.chest_data[index],self.label[index]