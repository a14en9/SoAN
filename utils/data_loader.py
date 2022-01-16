"""
Custom Dataset for Training
"""
#!/usr/bin/env python
import glob
import os
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import numpy as np
import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedKFold, KFold
import albumentations as A

"""
def fetch_loaders(processed_dir, batch_size=32, folder='train', shuffle=True, drop_last=True):
    
    if folder == 'train':
        train_dataset = GlacierDataset(os.path.join(processed_dir, folder))
        loader = {"train": DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle, drop_last=drop_last)}
    if folder == 'test':
        test_dataset = GlacierDataset_test(os.path.join(processed_dir,folder))
        loader = {"test": DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)}

    return loader
"""
def fetch_loaders(processed_dir, batch_size=32,
                  train_folder='train', dev_folder='dev', test_folder='test',
                  shuffle=True):
    train_dataset = GlacierDataset_train(os.path.join(processed_dir, train_folder))
    val_dataset = GlacierDataset_val(os.path.join(processed_dir, dev_folder))
    loader = {
        "train": DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=8, shuffle=shuffle, drop_last=True),
        "val": DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=8, shuffle=False, drop_last=False)
                      }

    if test_folder:
        test_dataset = GlacierDataset_test(os.path.join(processed_dir, test_folder))
        loader["test"] = DataLoader(test_dataset, batch_size=batch_size,
                                    num_workers=8, shuffle=False, drop_last=False)

    return loader


def splitter(processed_dir, n_splits):
    img_files = glob.glob(os.path.join(processed_dir, '*img*'))
    mask_files = [s.replace("img", "mask") for s in img_files]
    splitter = KFold(n_splits=n_splits, shuffle=True)
    splits = []
    for train_idx, test_idx in splitter.split(img_files, mask_files):
        splits.append((train_idx, test_idx))

    return splits

class DataSampler(Sampler):
    def __init__(self,
                 sample_idx,
                 data_source='../input/cassava-leaf-disease-classification/train.csv'):
        super().__init__(data_source)
        self.sample_idx = sample_idx
        self.df_images = pd.read_csv(data_source)

    def __iter__(self):
        image_ids = self.df_images['image_id'].loc[self.sample_idx]
        return iter(image_ids)

    def __len__(self):
        return len(self.sample_idx)


class DataBatchSampler(BatchSampler):
    def __init__(self,
                 sampler,
                 aug_count=5,
                 batch_size=30,
                 drop_last=True):
        super().__init__(sampler, batch_size, drop_last)
        self.aug_count = aug_count
        assert self.batch_size % self.aug_count == 0, 'Batch size must be an integer multiple of the aug_count.'

    def __iter__(self):
        batch = []

        for image_id in self.sampler:
            for i in range(self.aug_count):
                batch.append(image_id)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

def create_split_loaders(dataset, split, aug_count, batch_size):
    train_folds_idx = split[0]
    valid_folds_idx = split[1]
    train_sampler = DataSampler(train_folds_idx)
    valid_sampler = DataSampler(valid_folds_idx)
    train_batch_sampler = DataBatchSampler(train_sampler,
                                            aug_count,
                                            batch_size)
    valid_batch_sampler = DataBatchSampler(valid_sampler,
                                            aug_count=1,
                                            batch_size=batch_size,
                                            drop_last=False)
    train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler)
    valid_loader = DataLoader(dataset, batch_sampler=valid_batch_sampler)
    return (train_loader, valid_loader)

def get_all_split_loaders(dataset, cv_splits, aug_count=5, batch_size=30):
    """Create DataLoaders for each split.

    Keyword arguments:
    dataset -- Dataset to sample from.
    cv_splits -- Array containing indices of samples to
                 be used in each fold for each split.
    aug_count -- Number of variations for each sample in dataset.
    batch_size -- batch size.

    """
    split_samplers = []

    for i in range(len(cv_splits)):
        split_samplers.append(
            create_split_loaders(dataset,
                                 cv_splits[i],
                                 aug_count,
                                 batch_size)
        )
    return split_samplers


class GlacierDataset_train(Dataset):
    """Custom Dataset for Glacier Data

    Indexing the i^th element returns the underlying image and the associated
    binary mask

    """

    def __init__(self, folder_path):
        """Initialize dataset.

        Args:
            folder_path(str): A path to data directory

        """

        self.img_files = glob.glob(os.path.join(folder_path, '*img*'))
        self.mask_files = [s.replace("img", "mask") for s in self.img_files]
        self.transform = A.Compose([
            A.Flip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.RandomRotate90(p=0.5),
        ])

    def __getitem__(self, index):

        """ getitem method to retrieve a single instance of the dataset

        Args:
            index(int): Index identifier of the data instance

        Return:
            data(x) and corresponding label(y)
        """

        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = np.load(img_path)
        mask = np.load(mask_path)

        ###   flexiable to the affine transformation of images and masks as data augmentation
        # return torch.from_numpy(image).float(), torch.from_numpy(mask).float()
        transformed = self.transform(image=image, mask=mask)
        return torch.from_numpy(transformed['image']).float(), torch.from_numpy(transformed['mask']).float()


    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)

        """
        return len(self.img_files)


class GlacierDataset_val(Dataset):
    """Custom Dataset for Glacier Data

    Indexing the i^th element returns the underlying image and the associated
    binary mask

    """

    def __init__(self, folder_path):
        """Initialize dataset.

        Args:
            folder_path(str): A path to data directory

        """

        self.img_files = glob.glob(os.path.join(folder_path, '*img*'))
        self.mask_files = [s.replace("img", "mask") for s in self.img_files]

    def __getitem__(self, index):

        """ getitem method to retrieve a single instance of the dataset

        Args:
            index(int): Index identifier of the data instance

        Return:
            data(x) and corresponding label(y)
        """

        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = np.load(img_path)
        label = np.load(mask_path)

        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)

        """
        return len(self.img_files)

class GlacierDataset_test(Dataset):
    """Custom Dataset for Glacier Data

    Indexing the i^th element returns the underlying image and the associated
    binary mask

    """

    def __init__(self, folder_path):
        """Initialize dataset.

        Args:
            folder_path(str): A path to data directory

        """

        self.img_files = glob.glob(os.path.join(folder_path, '*img*'))
        self.mask_files = [s.replace("img", "mask") for s in self.img_files]

    def __getitem__(self, index):

        """ getitem method to retrieve a single instance of the dataset

        Args:
            index(int): Index identifier of the data instance

        Return:
            data(x) and corresponding label(y)
        """

        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = np.load(img_path)
        label = np.load(mask_path)

        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)

        """
        return len(self.img_files)

if __name__ == '__main__':
    # dataset_splitters = splitter('../patches/splits/train/', 5)
    loaders = fetch_loaders(str("../patches/splits/"), 32, shuffle=True)
    print(loaders.shape)