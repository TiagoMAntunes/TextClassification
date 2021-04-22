import torch


class SST(torch.utils.data.Dataset):
    def __init__(self, content, labels, mask):
        self.content= content
        self.labels = labels
        self.mask = mask

    def __getitem__(self, idx):
        return self.content[idx], self.labels[idx], self.mask[idx]

    def __len__(self):
        return self.content.shape[0]