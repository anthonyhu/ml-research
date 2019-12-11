import os
import zipfile
from glob import glob

import torch
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import Dataset

from utils import download_file_from_google_drive

GDRIVE_HASH = '10yVzRsnBnvsuhbGVg54IFSKTWhVAwBxr'
DATASET_DIR = './cifar10/dataset'


class CifarDataset(Dataset):
    """Classification dataset: input is an RGB image, label is an integer.
    The data has the following folder structure:

    root
       |__train
          |_____image_000000.png
          |_____label_000000.txt
          |_____image_000001.png
          |_____label_000001.txt
          |_____...
       |__val
          |_____image_000000.png
          |_____label_000000.txt
          |_____image_000001.png
          |_____label_000001.txt
          |_____...
    """
    def __init__(self, mode='train'):
        self.mode = mode
        self.augmentation = None

        # Download dataset
        if not os.path.isdir(os.path.join(DATASET_DIR, mode)):
            print('Dowloading CIFAR10 dataset..')
            zip_filename = 'tmp.zip'
            download_file_from_google_drive(GDRIVE_HASH, zip_filename)
            # Unzip train and val files
            with zipfile.ZipFile(zip_filename, 'r') as zip_file:
                zip_file.extractall()
                print(f'CIFAR10 data downloaded in {DATASET_DIR}.\n')
            # Delete zip file
            os.remove(zip_filename)

        self.filenames = dict()
        self.filenames['image'] = sorted(glob(os.path.join(DATASET_DIR, mode, 'image_*.png')))
        self.filenames['label'] = sorted(glob(os.path.join(DATASET_DIR, mode, 'label_*.txt')))

        assert len(self.filenames['image']) == len(self.filenames['label']), \
            'Mismatch in the size of input images and labels.'

    def __getitem__(self, index):
        batch = dict()
        # Load PIL image and label
        batch['image'] = Image.open(self.filenames['image'][index])
        with open(self.filenames['label'][index]) as f:
            batch['label'] = torch.tensor(int(f.readline()[0]))

        # Apply data augmentation
        if self.mode == 'train' and self.augmentation:
            batch['image'] = self.augmentation(batch['image'])
        # Convert to tensor
        batch['image'] = transforms.ToTensor()(batch['image'])

        return batch

    def __len__(self):
        return len(self.filenames['image'])
