import os
import os.path as osp

import torch
from torch.utils.data import Dataset
from PIL import Image
from group_capsules.datasets.utils import makedirs, download_url, extract_zip


class RotMNIST(Dataset):
    url = ('http://www.iro.umontreal.ca/~lisa/icml2007data/' +
           'mnist_rotation_new.zip')

    def __init__(self, root, split='train', transform=None):
        assert split in ['train', 'val', 'test']

        self.root = osp.expanduser(osp.normpath(root))
        self.split = split
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        self.transform = transform

        self.download()
        self.process()

        name = self.processed_files[['train', 'val', 'test'].index(split)]
        self.data, self.target = torch.load(name)

    @property
    def raw_files(self):
        name = 'mnist_all_rotation_normalized_float_{}.amat'
        files = [name.format(x) for x in ['train_valid', 'test']]
        return [osp.join(self.raw_dir, file) for file in files]

    @property
    def processed_files(self):
        dir = self.processed_dir
        return [osp.join(dir, f) for f in ['training.pt', 'val.pt', 'test.pt']]

    def __getitem__(self, i):
        img, target = self.data[i], self.target[i]
        img = Image.fromarray(img.numpy(), mode='F')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data.size(0)

    def download(self):
        if all([osp.exists(f) for f in self.raw_files]):
            return

        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        if all([osp.exists(f) for f in self.processed_files]):
            return

        print('Processing...')

        makedirs(self.processed_dir)
        data, target = self._process(self.raw_files[0])
        torch.save((data, target), self.processed_files[0])
        torch.save((data[10000:], target[10000:]), self.processed_files[1])
        torch.save(self._process(self.raw_files[1]), self.processed_files[2])

        print('Done!')

    def _process(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')[:-1]
            data = [[float(x) for x in line.split()] for line in lines]
        data = torch.tensor(data)
        target = data[:, -1].long().squeeze().contiguous()
        data = data[:, :-1].view(-1, 28, 28).contiguous() * 255

        return data, target
