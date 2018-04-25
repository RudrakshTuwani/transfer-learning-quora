from torch.utils.data import Dataset


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