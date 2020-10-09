import torch


class DataLoader:
    
    def __init__(self,dataset,batch_size=1,shuffle=False):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)

    def __iter__(self):
        self._i = 0

        if self.shuffle:
            index_shuffle = torch.randperm(self.data_size)
            self.dataset = [ self.dataset[v] for v in index_shuffle ]

        return self

    def __next__(self):

        i1 = self.batch_size * self._i
        i2 = min( self.batch_size * ( self._i + 1 ), self.data_size )

        if i1 >= self.data_size:
            raise StopIteration()

        value = self.dataset[i1:i2]
        self._i += 1

        return value