import os
import json
import tqdm

import torch
import numpy as np
import sys

from utils import utils
from datasets.common import get_dataloader, maybe_dictionarize
from models.heads import get_classification_head
from models.modeling import ImageClassifier

from datasets.registry import get_dataset
import torchvision.transforms as transforms

def calculate_stats(model, dataset_name, args,model_name, save_dir=None, save_stat_dir=None):
    
    
    model.eval()
    model.cuda()
    if save_stat_dir is not None:
        os.makedirs(save_stat_dir, exist_ok=True)
    stat_dict = {}

    preprocess_fn = transforms.Compose(model.val_preprocess.transforms[:-1]) if args.post_normalize else model.val_preprocess
    
    dataset = get_dataset(
        args.train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        is_transfer=True,
        use_train_for_proxy=args.use_train_for_proxy,
        data_ratio=args.data_ratio,
        args=args,
    )
    data_loader = get_dataloader(
            dataset,is_train=True, args=args, image_encoder=None, is_transfer=True)
    device = args.device

    logit_list=[]   
    # count = 0
    with torch.no_grad():
        
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            data = maybe_dictionarize(data)
            if args.use_precompute_features:
                x = None
                source_feats = data[f'{model.image_encoder.model_name}_features'].to(device)
                source_feats /= source_feats.norm(dim=-1, keepdim=True)
                logits = model.classification_head(source_feats)
            else:
                x = data['images'].to(device)
                # y = data['labels'].to(device)

                logits = utils.get_logits(x, model)
            
            # n += y.size(0)
            # if save_stat_dir is not None:
            #     stat_dict.update({f"{i}_pred":logits.cpu().tolist(),
            #                          f"{i}_gt":y.cpu().tolist()})
            
            logit_list.append(logits)
    # class_wise_logit_
    logit_list = torch.cat(logit_list, dim=0)
    class_wise_logit_mean = logit_list.mean(dim=0)
    class_wise_logit_std = logit_list.std(dim=0)
    
    all_logit_mean = logit_list.mean().item()
    all_logit_std = logit_list.std().item()
    stat_dict = {'class_wise_logit_mean':class_wise_logit_mean,
                 'class_wise_logit_std':class_wise_logit_std,
                 'all_logit_mean':all_logit_mean,
                 'all_logit_std':all_logit_std,
                 'model_name':model_name,}
    
    print(f"Class-wise logit mean: {class_wise_logit_mean} \n")
    print(f"Class-wise logit std: {class_wise_logit_std} \n")
    print(f"All logit mean: {all_logit_mean} \n")
    print(f"All logit std: {all_logit_std} \n")
    
    if save_stat_dir is not None:
        filename = os.path.join(save_stat_dir, f'stat_dict.json')
        with open(filename, 'w') as f:
            json.dump(stat_dict, f)
    
    return stat_dict