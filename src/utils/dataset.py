from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import csv
import os


class CaptchaDataset(Dataset):
    def __init__(self, root, split='train', transform=None, seed=9632):
        self.root = root
        self.transform = transform
        self.split = split  # train, valid, or test
        self.seed = seed
        # if train or valid, load metadata.csv
        if self.split in ['train', 'valid']:
            self.labelPathPair = self.loadMetaData()
            # train valid split
            np.random.seed(self.seed)
            randIdxs = np.random.permutation(len(self.labelPathPair))
            trainIdxSplit = int(0.8 * len(randIdxs))
            if self.split == 'train':
                randIdxs = randIdxs[:trainIdxSplit]
            else:
                randIdxs = randIdxs[trainIdxSplit:]
            self.labelPathPair = [self.labelPathPair[i] for i in randIdxs]
        else:
            # test split, no labels
            self.labelPathPair = [(None, f) for f in os.listdir(os.path.join(self.root, 'test'))]

    def loadMetaData(self):
        # load metadata
        labelPathPair = []
        with open(os.path.join(self.root, 'train_metadata.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                labelPathPair.append((row[0], os.path.join(self.root, 'train', row[1])))
        return labelPathPair

    def __len__(self):
        return len(self.labelPathPair)

    def __getitem__(self, idx):
        label, path = self.labelPathPair[idx]
        img = read_image(path)
        if self.transform:
            img = self.transform(img)
        return img, label