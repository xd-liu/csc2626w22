from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
import os
import re

class DrivingDataset(Dataset):
    
    def __init__(self, root_dir, categorical = False, classes=-1, transform=None):
        """
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in listdir(self.root_dir) if f.endswith('jpg')]
        self.categorical = categorical
        self.classes = classes

        self.class_dict = self.get_class_dict()
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        basename = self.filenames[idx]
        img_name = os.path.join(self.root_dir, basename)
        image = io.imread(img_name)

        m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', basename)
        steering_command = np.array(float(m.group(3)), dtype=np.float32)

        if self.categorical:
            steering_command = int(((steering_command + 1.0)/2.0) * (self.classes - 1)) 
            
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'cmd': steering_command}
    
    def get_class_dict(self):
        class_dict = {'straight': [],
                            'turn': []}

        straight_command = int(((0. + 1.0)/2.0) * (self.classes - 1))
        for i, filename in enumerate(self.filenames):
            m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', filename)
            steering_command = np.array(float(m.group(3)), dtype=np.float32)
            steering_command = int(((steering_command + 1.0)/2.0) * (self.classes - 1))

            if steering_command == straight_command:
                class_dict['straight'].append(i)
            else:
                class_dict['turn'].append(i)
        
        return class_dict


class Balanced_Sampler(Sampler):
    def __init__(self, data_source, class_dict):
        self.data_source = data_source
        self.class_dict = class_dict
        self.balance_list = self.generate_balance_list()

    def __iter__(self):
        return iter(self.balance_list)

    def __len__(self):
        return len(self.balance_list)
    
    def generate_balance_list(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        straight_indices = torch.tensor(self.class_dict['straight'])
        turn_indices = torch.tensor(self.class_dict['turn'])

        straight_indices = straight_indices[torch.randperm(straight_indices.size(0), generator=generator)].tolist()
        turn_indices = turn_indices[torch.randperm(turn_indices.size(0), generator=generator)].tolist()

        BATCH_SIZE = 256
        straight_size = int(BATCH_SIZE * (len(straight_indices) / len(self.data_source)))
        turn_size = BATCH_SIZE - straight_size

        balance_list = []
        straight_pointer = 0
        turn_pointer = 0
        while straight_pointer + straight_size < len(straight_indices) and \
            turn_pointer + turn_size < len(turn_indices):

            balance_list += straight_indices[straight_pointer:straight_pointer + straight_size]
            balance_list += turn_indices[turn_pointer:turn_pointer + turn_size]

            straight_pointer += straight_size
            turn_pointer += turn_size
        
        balance_list += straight_indices[straight_pointer:]
        balance_list += turn_indices[turn_pointer:]

        return balance_list

        
