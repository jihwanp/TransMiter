import os
import torch
import torchvision.datasets as datasets
import random 

class MNIST:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 is_transfer=False,
                 use_train_for_proxy=False, 
                 data_ratio=1.0,
                 args=None):


        self.train_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=True,
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=False,
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=is_transfer,
            num_workers=num_workers
        )
        
        if data_ratio < 1.0:
            
            random.seed(42)
            which_data = self.train_dataset if use_train_for_proxy else self.test_dataset
            random_indexs = random.sample(range(len(which_data)), int(data_ratio * len(which_data)))
            print('data_ratio:' + str(data_ratio))
            print('random_indexs:'+str(random_indexs))
            self.proxy_training_data = torch.utils.data.Subset(which_data,
                                                                 random_indexs)
        else:
            self.proxy_training_data = self.train_dataset if use_train_for_proxy else self.test_dataset

        self.proxy_loader_shuffle = torch.utils.data.DataLoader(
            self.proxy_training_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']