import os
import torch
import torchvision.datasets as datasets
import random
from datasets.feature_data import FeatureDataset

class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 is_transfer=False,
                 use_train_for_proxy=False, 
                 data_ratio=1.0,
                 args=None):
        # Data loading code
        traindir = os.path.join(location, 'sun397', 'train')
        valdir = os.path.join(location, 'sun397', 'val')


        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=is_transfer,
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
        
        feat_dir = os.path.join(location, 'features/SUN397/naive')
        
        train_feats = FeatureDataset(feat_dir,'train')
        test_feats = FeatureDataset(feat_dir,'test')
        self.train_feats_loader = torch.utils.data.DataLoader(train_feats, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        self.test_feats_loader = torch.utils.data.DataLoader(test_feats, batch_size=batch_size, shuffle=False,num_workers=num_workers)
        
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]
