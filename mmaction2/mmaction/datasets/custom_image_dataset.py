from torch.utils.data import Dataset

class CustomImageDataSet(Dataset):
    def __init__(self,frames):
        self.frames=frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,idx):
        return self.frames[idx]
