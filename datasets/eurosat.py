import os
import torch
import torchvision.datasets as datasets
import re
import random

def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out

class EuroSATBase:
    def __init__(self,
                 preprocess,
                 test_split,
                 location='~/datasets',
                 batch_size=32,
                 num_workers=16,
                 is_transfer=False,
                 use_train_for_proxy=False, 
                 data_ratio=1.0,
                 args=None):
        # Data loading code
        traindir = os.path.join(location, 'EuroSAT_splits', 'train')
        testdir = os.path.join(location, 'EuroSAT_splits', test_split)


        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(testdir, transform=preprocess)
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
        
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        self.classnames = [pretify_classname(c) for c in self.classnames]
        ours_to_open_ai = {
            'annual crop': 'annual crop land',
            'forest': 'forest',
            'herbaceous vegetation': 'brushland or shrubland',
            'highway': 'highway or road',
            'industrial area': 'industrial buildings or commercial buildings',
            'pasture': 'pasture land',
            'permanent crop': 'permanent crop land',
            'residential area': 'residential buildings or homes or apartments',
            'river': 'river',
            'sea lake': 'lake or sea',
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]


class EuroSAT(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/datasets',
                 batch_size=32,
                 num_workers=16,
                 is_transfer=False,
                 use_train_for_proxy=False, 
                 data_ratio=1.0,
                 args=None):
        super().__init__(preprocess, 'test', location, batch_size, num_workers,is_transfer,use_train_for_proxy,data_ratio)


class EuroSATVal(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/datasets',
                 batch_size=32,
                 num_workers=16,
                 is_transfer=False,
                 use_train_for_proxy=False,
                 data_ratio=1.0,
                 args=None):
        super().__init__(preprocess, 'val', location, batch_size, num_workers, is_transfer, use_train_for_proxy, data_ratio)
