import os
import torch
import torchvision.datasets as datasets
import random
from datasets.feature_data import FeatureDataset, FewshotFeatureDataset, DGFeatureDataset

data_name_dict={
    'Caltech':'Caltech101',
    'EuroSAT':'EuroSAT',
    'Food101':'Food101',
    'SUN397':'SUN397',
    'Aircraft':'FGVCAircraft',
    'ImageNet':'ImageNet',
    'Cars':'StanfordCars',
    'Pets':'OxfordPets',
    'DTD':'DescribableTextures',
    'Flowers':'OxfordFlowers',
    'UCF':'ImageUCF101',
    'ImageNet_a':'ImageNetA',
    'ImageNet_r':'ImageNetR',
    'ImageNet_sketch':'ImageNetSketch',
    'ImageNetv2':'ImageNetV2',
    
}

b2n_data_name_dict={
    'Caltech':'caltech101',
    'EuroSAT':'eurosat',
    'Food101':'food101',
    'SUN397':'sun397',
    'Aircraft':'fgvc_aircraft',
    'ImageNet':'imagenet',
    'Cars':'stanford_cars',
    'Pets':'oxford_pets',
    'DTD':'dtd',
    'Flowers':'oxford_flowers',
    'UCF':'ucf101',
    'ImageNet_a':'imagenet_a',
    'ImageNet_r':'imagenet_r',
    'ImageNet_sketch':'imagenet_sketch',
    'ImageNetv2':'imagenetv2',
    
}

class FewshotDataset:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 is_transfer=False,
                 use_train_for_proxy=False, 
                 data_ratio=1.0,
                 args=None,
                 dg_type=None,
                 post_train=False):
        if post_train:
            if args.setting == 'few_shot':
                feat_dirs = []
                
                d_folder_name = b2n_data_name_dict[args.dataset]
                feat_dir = os.path.join(location,f'fewshot_zs/{d_folder_name}/{args.seed}/')
                train_feats = FewshotFeatureDataset([feat_dir],'train',args,post_train=True)
            elif args.setting == 'base2novel':
                feat_dir = os.path.join(location,f'base2new_zs/{b2n_data_name_dict[args.dataset]}/{args.seed}/')
                train_feats = FewshotFeatureDataset([feat_dir],'train',args,post_train=True)
            elif args.setting == 'cross_data':
                raise NotImplementedError
            elif args.setting == 'dg':
                # # feat_dirs = os.path.join(location,f'dg_zs/{args.dg_dataset}/{args.seed}/')
                # raise NotImplementedError
                d_folder_name = b2n_data_name_dict[args.dataset]
                feat_dir = os.path.join(location,f'fewshot_zs/{d_folder_name}/{args.seed}/')
                train_feats = FewshotFeatureDataset([feat_dir],'train',args,post_train=True)
        # else:
        if args.setting == 'few_shot':
            # feat_dir = os.path.join(location,f'fewshot_features_v2/{data_name_dict[args.dataset]}/{args.seed}/{args.num_shot}/')
            feat_dirs = []
            
            for ft_st in args.ft_strategy:
                d_folder_name = data_name_dict[args.dataset] if 'features' in ft_st else b2n_data_name_dict[args.dataset]
                feat_dir = os.path.join(location,f'fewshot_{ft_st}/{d_folder_name}/{args.seed}/{args.num_shot}/')
                feat_dirs.append(feat_dir)
                # feat_dirs = [os.path.join(location,f'fewshot_{ft_st}/{b2n_data_name_dict[args.dataset]}/{args.seed}/{args.num_shot}/') for ft_st in args.ft_strategy]
            # train_feats = FewshotFeatureDataset(feat_dirs,'all_train',args) if args.use_all_train else FewshotFeatureDataset(feat_dirs,'train',args)
            # if not post_train:
            if not post_train:
                train_feats = FewshotFeatureDataset(feat_dirs,'train',args) if not args.dg else FewshotFeatureDataset(feat_dirs,'test',args)
        elif args.setting == 'base2novel':
            feat_dirs = [os.path.join(location,f'base2new_{ft_st}/{b2n_data_name_dict[args.dataset]}/{args.seed}/{args.num_shot}/') for ft_st in args.ft_strategy]
                # feat_dir = os.path.join(location,f'base2new_features/{b2n_data_name_dict[args.dataset]}/{args.seed}/16/')
            if not post_train:
                train_feats = FewshotFeatureDataset(feat_dirs,'train',args)
            self.base_class_idx = train_feats.base_class_idx
            self.novel_class_idx = train_feats.novel_class_idx
        elif args.setting == 'cross_data':
            # feat_dir = os.path.join(location,f'crossdata/{b2n_data_name_dict[args.dataset]}/{args.seed}/16/')
            feat_dirs = [os.path.join(location,f'crossdata_{ft_st}/{b2n_data_name_dict[args.dataset]}/{args.seed}/{args.num_shot}/') for ft_st in args.ft_strategy]
            if not post_train:
                train_feats = FewshotFeatureDataset(feat_dirs,'train',args)
        elif args.setting == 'dg':
            feat_dirs = [os.path.join(location,f'dg_{ft_st}/imagenet/{args.seed}/{args.num_shot}/') for ft_st in args.ft_strategy]
            if not post_train:
                train_feats = FewshotFeatureDataset(feat_dirs,'train',args)
            
        else:
            raise NotImplementedError
            
        # test_feats = FewshotFeatureDataset([feat_dirs[0]],'test',args)
        if args.setting=='dg':
            
            # for dg_eval in ['','_a','_r','_sketch','v2']:
            feat_dirs = [os.path.join(location,f'dg_{ft_st}/{args.dg_dataset}/{args.seed}/{args.num_shot}/') for ft_st in args.ft_strategy]
            test_feats = DGFeatureDataset(feat_dirs,'test',train_feats.class_name,args)
            self.dg_class_idx = test_feats.dg_class_idx
        else:
            
            test_feats = FewshotFeatureDataset(feat_dirs,'test',args)
    
        if args.setting !='dg':
            for train_class , test_class in zip(train_feats.class_name,test_feats.class_name):
                assert train_class == test_class, f"train class {train_class} != test class {test_class}"
            
        self.classnames = train_feats.class_name
        
        if args.data_ratio < 1.0:
                
            random.seed(42)
            random_indexs = random.sample(range(len(train_feats)), int(args.data_ratio * len(train_feats)))
            print('data_ratio:' + str(args.data_ratio))
            # print('random_indexs:'+str(random_indexs))
            train_feats = torch.utils.data.Subset(train_feats, random_indexs)
        
        # self.class_names = [name.replace('_',' ') for name in self.classnames]
        
        # self.train_feats_loader = torch.utils.data.DataLoader(train_feats, batch_size=batch_size, shuffle=True,num_workers=num_workers, drop_last=args.loader_ver2)
        self.train_feats_loader = torch.utils.data.DataLoader(train_feats, batch_size=batch_size, shuffle=True,num_workers=num_workers, drop_last=False)
            
        self.test_feats_loader = torch.utils.data.DataLoader(test_feats, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    
        if args.loader_ver2:
            # self.transfer_feats_loader = torch.utils.data.DataLoader(train_feats, batch_size=1024, shuffle=False,num_workers=num_workers)
            self.transfer_feats_loader = self.train_feats_loader
        
        # idx_to_class = dict((v, k)
        #                     for k, v in self.train_dataset.class_to_idx.items())
        # self.classnames = [idx_to_class[i].replace(
        #     '_', ' ') for i in range(len(idx_to_class))]