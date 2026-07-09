# Custom Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class ADNIDatasetDANN(Dataset):
    def __init__(self, features_list, sex_list, age_list, domain_list):
        self.features = features_list
        self.sex = sex_list
        self.age = age_list
        self.domain = domain_list

    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.sex[idx], dtype=torch.float32),
            torch.tensor(self.age[idx], dtype=torch.float32),
            torch.tensor(self.domain[idx], dtype=torch.long),
        )

class ADNIDataset(Dataset):
    def __init__(self, features_list, sex_list, age_list):
        self.features = features_list
        self.sex = sex_list
        self.age = age_list

    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.sex[idx], dtype=torch.float32),
            torch.tensor(self.age[idx], dtype=torch.float32),
        )

class ADNIDatasetLite(Dataset):

    def __init__(self,image_list, sex_list, age_list):
        self.images = image_list
        self.sex = sex_list
        self.age = age_list

    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.sex[idx],
            self.age[idx],
        )


class ADNIDatasetViT(Dataset):
    def __init__(self,image_list, age_list):
        self.images = image_list
        self.label = age_list

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.label[idx],
        )
    
class ADNIDatasetViTDANN(Dataset):
    def __init__(self,image_list, age_list, domain_list):
        self.images = image_list
        self.label = age_list
        self.domain = domain_list

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.label[idx],
            self.domain[idx],
        )

class ADNIGenderedDatasetViT(Dataset):
    def __init__(self, image_list, age_list, sex_list):
        self.images = image_list
        self.ages = age_list
        self.sexes = sex_list

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        return (
            self.images[idx],   # filepath to NIfTI
            (self.ages[idx], self.sexes[idx])  # return tuple (age, sex)
        )