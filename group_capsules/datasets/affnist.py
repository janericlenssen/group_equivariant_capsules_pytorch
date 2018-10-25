import os
import os.path as osp
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.io.matlab import loadmat
from group_capsules.datasets.utils import makedirs, download_url, extract_zip


class AffNIST(Dataset):
    url = 'http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed'
    files = ['training_and_validation_batches', 'test_batches']

    def __init__(self, root, train=True, transform=None):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        self.transform = transform

        self.download()
        self.process()

        name = self.processed_files[0] if train else self.processed_files[1]
        self.data, self.target = torch.load(name)

    @property
    def raw_files(self):
        return [osp.join(self.raw_dir, f) for f in self.files]

    @property
    def processed_files(self):
        folder = self.processed_dir
        return [osp.join(folder, f) for f in ['training.pt', 'test.pt']]

    def __getitem__(self, i):
        img, target = self.data[i], self.target[i]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data.size(0)

    def download(self):
        if all([osp.exists(f) for f in self.raw_files]):
            return

        for f in self.files:
            path = download_url('{}/{}.zip'.format(self.url, f), self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        if all([osp.exists(f) for f in self.processed_files]):
            return

        print('Processing...')

        makedirs(self.processed_dir)
        torch.save(self._process(self.raw_files[0]), self.processed_files[0])
        torch.save(self._process(self.raw_files[1]), self.processed_files[1])

        print('Done!')

    def _process(self, folder):
        data, target = [], []
        for f in sorted(glob.glob('{}/*.mat'.format(folder))):
            f = loadmat(f)['affNISTdata'][0][0]
            data += [torch.from_numpy(f[2]).t().view(-1, 40, 40)]
            target.append(torch.from_numpy(f[5]).squeeze())
        return torch.cat(data, dim=0).contiguous(), torch.cat(target, dim=0)
