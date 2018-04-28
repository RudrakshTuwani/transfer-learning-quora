from torch.utils.data import Dataset
from fastai.text import A

class DatasetStream(Dataset):
    def __init__(self, embed_array, labels_array, idxs):
        self.x = embed_array
        self.y = labels_array
        self.idxs = idxs
        
    def __getitem__(self, i):
        idx = self.idxs[i]
        return [self.x[:, idx, :], self.y[idx]]
            
    def __len__(self):
        return len(self.idxs)

class PairedDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
        
    def __getitem__(self, i):
        return A(self.x[i, 0], self.x[i, 1]), self.y[i]
    
    def __len__(self):
        return self.y.shape[0]