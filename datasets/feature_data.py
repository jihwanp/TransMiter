import os
import torch
from torch.utils.data import Dataset, DataLoader


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
    
}

class FeatureDataset(Dataset):
    def __init__(self, feature_dir='', mode='train'):
        feature_file = f"{feature_dir}/{mode}_features.pt"
        self.data = torch.load(feature_file, map_location='cpu')
        self.keys = list(self.data.keys())
    def __len__(self):
        
        return len(self.data['labels'])

    def __getitem__(self, idx):
        return {k:self.data[k][idx] for k in self.keys}
    
class FewshotFeatureDataset(Dataset):
    def __init__(self, feature_dirs=[], mode='train',args=None, post_train=False):
        self.mode = mode
        self.feature_dirs = feature_dirs
        self.data_dict = {'labels':[],
                          'label_names':[],}
        if "__pretrained__" in args.source_model:
            source_model,pretrained = args.source_model.split("__pretrained__")
        else:
            source_model = args.source_model
        self.data_dict[f'{source_model}_features'] = []
        if not post_train:
            if args.use_ft_logits:
                self.data_dict[f'{source_model}_ft_logits'] = []
            else:
                self.data_dict[f'{source_model}_ft_features'] = []

        if "__pretrained__" in args.target_model:
            target_model,pretrained = args.target_model.split("__pretrained__")
        else:
            target_model = args.target_model
        self.data_dict[f'{target_model}_features'] = []
        
        if not post_train:
            if args.use_ft_logits:
                self.data_dict[f'{target_model}_ft_logits'] = []
            
            else:
                
                self.data_dict[f'{target_model}_ft_features'] = []
        
        if len(args.intermediate_models)>0:
            for target_model in args.intermediate_models:
                self.data_dict[f'{target_model}_features'] = []
        
        if args.setting == 'base2novel':
            self.data_dict['base'] = []
            self.data_dict['novel'] = []
        
        for dir in feature_dirs:
            if 'features' in dir and mode=='train':
                data = torch.load(os.path.join(dir,f'all_{mode}_features.pt'), map_location='cpu')
            else:
                data = torch.load(os.path.join(dir,f'{mode}_features.pt'), map_location='cpu')
            if post_train:
                data
            for k in self.data_dict.keys():
                if k in data.keys():
                    self.data_dict[k].append(data[k])
                
                    
                # elif '_ft_' not in k and 'features' in k:
                #     zeroshot_dir = f'all_features/{b2n_data_name_dict[args.dataset]}/1/16/{model}__pretrained__{pretrained}/features/pretrained/{mode}_zero_shot_features.pt'
                #     zs_data = torch.load(os.path.join(args.data_location,zeroshot_dir), map_location='cpu')
                #     self.data_dict[k].append(zs_data[k])
                # elif '_ft_' in k and 'logits' in k:
                #     self.data_dict[k].append(data[k.replace('_ft_logits','_logits')])
        
        if args.setting == 'base2novel':
            first_base_idx = self.data_dict['base'][0]
            first_novel_idx = self.data_dict['novel'][0]
            
            for bb in self.data_dict['base']:
                assert (first_base_idx == bb)
            
            for nn in self.data_dict['novel']:
                assert (first_novel_idx == nn)
        
        # self.data_dict['label_names'] = [dict(sorted(d.items())) for d in self.data_dict['label_names']]
        
        first_label_names = self.data_dict['label_names'][0]
        for label_names in self.data_dict['label_names']:
            assert (first_label_names == label_names)
        
        self.keys = list(self.data_dict.keys())
        class_name  = self.data_dict['label_names'][0]
        self.class_name = [v.replace('_',' ') for k,v in sorted(class_name.items())]
        self.num_classes = len(self.class_name)
    
        if args.setting == 'base2novel':
            assert self.num_classes == len(first_base_idx)+len(first_novel_idx), f"num sample {self.num_classes} != num base {len(first_base_idx)} + num novel {len(first_novel_idx)}"
            self.base_class_idx = first_base_idx
            self.novel_class_idx = first_novel_idx
        
        for k in self.data_dict.keys():

            if 'ft' in k and args.target_model.split('__pretrained__')[0] in k:
                # if len(self.data_dict[k])==0 and 'logits' not in k:
                if len(self.data_dict[k])==0 or len(self.data_dict[k])<len(self.data_dict['labels']):
                    print(f"length of {k} is {len(self.data_dict[k])}, less than {len(self.data_dict['labels'])}")
                    print(f"pad with last one of {source_model} fine-tuned, this will not be used in training but evaluation will be problematic.. please fix it later")
                    for i in range(len(self.data_dict['labels'])-len(self.data_dict[k])):
                        self.data_dict[k].append(self.data_dict[k.replace(target_model,source_model)][-1])

        
                        
        if args.use_ft_logits:
            self.data_dict
        
        if post_train:
            self.data_dict
        
    def __len__(self):
        
        return len(self.data_dict['labels'][0])

    def __getitem__(self, idx):

        key_list = [k for k in self.keys if (k not in ['label_names','base','novel']) and ('text' not in k)]
        item = {k: [self.data_dict[k][i][idx] for i in range(len(self.feature_dirs))] if 'ft' in k else self.data_dict[k][0][idx] for k in key_list}
        return item

class DGFeatureDataset(Dataset):
    def __init__(self, feature_dirs=[], mode='train',orig_class_list=[],args=None):
        self.mode = mode
        self.feature_dirs = feature_dirs
        self.data_dict = {'labels':[],
                          'label_names':[],}
        if "__pretrained__" in args.source_model:
            model,pretrained = args.source_model.split("__pretrained__")
        else:
            model = args.source_model
        self.data_dict[f'{model}_features'] = []
        self.data_dict[f'{model}_ft_features'] = []
            
        
        if "__pretrained__" in args.target_model:
            model,pretrained = args.target_model.split("__pretrained__")
        else:
            model = args.target_model
        self.data_dict[f'{model}_features'] = []
        self.data_dict[f'{model}_ft_features'] = []
            
        if len(args.intermediate_models)>0:
            for model in args.intermediate_models:
                self.data_dict[f'{model}_features'] = []
        
        if args.setting == 'base2novel':
            self.data_dict['base'] = []
            self.data_dict['novel'] = []
        
        for dir in feature_dirs:
            if 'features' in dir and mode=='train':
                data = torch.load(os.path.join(dir,f'all_{mode}_features.pt'), map_location='cpu')
            else:
                data = torch.load(os.path.join(dir,f'{mode}_features.pt'), map_location='cpu')
            for k in self.data_dict.keys():
                if k in data.keys():
                    self.data_dict[k].append(data[k])
                elif '_ft_' not in k and 'features' in k:
                    zeroshot_dir = f'all_features/{b2n_data_name_dict[args.dataset]}/1/16/{model}__pretrained__{pretrained}/features/pretrained/{mode}_zero_shot_features.pt'
                    zs_data = torch.load(os.path.join(args.data_location,zeroshot_dir), map_location='cpu')
                    self.data_dict[k].append(zs_data[k])
        
        
        
        if args.setting == 'base2novel':
            first_base_idx = self.data_dict['base'][0]
            first_novel_idx = self.data_dict['novel'][0]
            
            for bb in self.data_dict['base']:
                assert (first_base_idx == bb)
            
            for nn in self.data_dict['novel']:
                assert (first_novel_idx == nn)
        
        # self.data_dict['label_names'] = [dict(sorted(d.items())) for d in self.data_dict['label_names']]
        
        first_label_names = self.data_dict['label_names'][0]
        for label_names in self.data_dict['label_names']:
            assert (first_label_names == label_names)
        
        self.keys = list(self.data_dict.keys())
        class_name  = self.data_dict['label_names'][0]
        self.class_name = [v.replace('_',' ') for k,v in sorted(class_name.items())]
        self.num_classes = len(self.class_name)
    
        if args.setting == 'base2novel':
            assert self.num_classes == len(first_base_idx)+len(first_novel_idx), f"num sample {self.num_classes} != num base {len(first_base_idx)} + num novel {len(first_novel_idx)}"
            self.base_class_idx = first_base_idx
            self.novel_class_idx = first_novel_idx

        for k in self.data_dict.keys():
            if 'ft' in k and args.target_model.split('__pretrained__')[0] in k:
                if len(self.data_dict[k])==0:
                    self.data_dict[k] = self.data_dict["".join(k.split('_ft'))]
        
        
        self.dg_class_idx = [orig_class_list.index(class_name) for class_name in self.class_name]
        
        
    def __len__(self):
        
        return len(self.data_dict['labels'][0])

    def __getitem__(self, idx):

        key_list = [k for k in self.keys if (k not in ['label_names','base','novel']) and ('text' not in k)]
        item = {k: [self.data_dict[k][i][idx] for i in range(len(self.feature_dirs))] if 'ft' in k else self.data_dict[k][0][idx] for k in key_list}
        return item

def get_datasets(args):
    train_dataset = FeatureDataset(f"{args.feature_dir}/train_features.pt")
    test_dataset = FeatureDataset(f"{args.feature_dir}/test_features.pt")
    return train_dataset, test_dataset

# Example usage:
# args = Namespace(feature_dir='/path/to/features')
# train_dataset, test_dataset = get_datasets(args)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    
    feature_dir = './data/features/DTD/naive'
    train_dataset = FeatureDataset(f"{feature_dir}")
    # import pdb;pdb.set_trace()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for i,batch in enumerate(train_loader):
        import pdb;pdb.set_trace()
        print(i)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # print(len(train_loader))
    # print