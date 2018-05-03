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
        return A(self.x[i, 0]), A(self.x[i, 1]), self.y[i]
    
    def __len__(self):
        return self.y.shape[0]
    
    
class PairedEmbedDataset(Dataset):
    def __init__(self, embed_array, labels_array):
        self.x, self.y = embed_array, labels_array
        
    def __getitem__(self, i):
        return (self.x[i], self.y[i])
    
    def __len__(self):
        return self.y.shape[0]

    
class USEEnsembleDataset(Dataset):
    def __init__(self, qn_toks, qn_lbl, use_embed):
        self.qn_toks, self.qn_lbl, self.use_embed = qn_toks, qn_lbl, use_embed

    def __getitem__(self, i):
        return A(self.qn_toks[i, 0]), A(self.qn_toks[i, 1]), A(self.use_embed[0, i, :]),\
               A(self.use_embed[1, i, :]), self.qn_lbl[i]
    
    def __len__(self):
        return self.qn_toks.shape[0]
    