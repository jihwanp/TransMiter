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
import torch.nn.functional as F

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def eval_baseline_table( source_zs_model,source_ft_model,target_zs_model,dataset_name, alpha, baseline_mode=None,args=None):
    # model.eval()
    source_zs_model.eval()
    if isinstance(source_ft_model, list):
        for m in source_ft_model:
            m.eval()
    else:
        source_ft_model.eval()
    target_zs_model.eval()
    
    dataset = get_dataset(
        dataset_name,
        # model.val_preprocess if not args.post_normalize else transforms.Compose(model.val_preprocess.transforms[:-1]),
        None,
        location=args.data_location,
        batch_size=512,
        args=args
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    with torch.no_grad():
        if args.setting == 'base2novel':
            top1_base, base_correct, base_n = 0., 0., 0.
            top1_novel, novel_correct, novel_n = 0., 0., 0.
            n=0
        else:
            top1, correct, n = 0., 0., 0.
        sharpness = 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            if args.use_precompute_features:
                x = None
                y = data['labels'].to(device=device)
                source_zs_inputs = data[f'{source_zs_model.image_encoder.model_name}_features'].to(device=device)
                source_zs_inputs/=source_zs_inputs.norm(dim=-1,keepdim=True)
                if args.setting =='dg':
                    source_zs_logits = source_zs_model.classification_head(source_zs_inputs)[:,:args.num_classes]
                else:
                    source_zs_logits = F.log_softmax(source_zs_model.classification_head(source_zs_inputs)[:,:args.num_classes],dim=-1)
                
                all_logits = [] 
                if args.use_ft_logits:
                    source_ft_inputs = data[f'{source_ft_model[0].image_encoder.model_name}_ft_logits']
                    all_logits = [] 
                    for m,inp in zip(source_ft_model,source_ft_inputs):
                        logit = inp.to(device=device)
                        # if m.image_encoder.model.logit_scale is not None:
                        #     logit_scale = m.image_encoder.model.logit_scale.exp()
                        #     logit = logit*logit_scale
                        # if m.image_encoder.model.logit_bias is not None:
                        #     logit_bias = m.image_encoder.model.logit_bias
                        #     logit = logit + logit_bias
                        all_logits.append(logit[:,:args.num_classes])
                    # source_ft_logits = torch.stack(all_logits).mean(dim=0)
                else:
                    source_ft_inputs = data[f'{source_ft_model[0].image_encoder.model_name}_ft_features']
                    
                    for m,inp in zip(source_ft_model,source_ft_inputs):
                        inp = inp.to(device=device)
                        inp/=inp.norm(dim=-1,keepdim=True)
                        logits = m.classification_head(inp)[:,:args.num_classes]
                        all_logits.append(logits)
                # source_ft_logits = torch.stack(all_logits).mean(dim=0)
                
                if args.apply_tf_ensemble:
                    source_ft_logits = torch.stack(all_logits)
                    prob = source_ft_logits.softmax(dim=-1)
                    max_prob_per_model = prob.max(dim=-1,keepdim=True)[0]
                    weight_per_model = max_prob_per_model.softmax(dim=0)
                    source_ft_logits = (weight_per_model*source_ft_logits).sum(dim=0)
                else:
                    source_ft_logits = torch.stack(all_logits).mean(dim=0)
                    
                if args.setting =='dg':
                    source_ft_logits = source_ft_logits
                else:
                    source_ft_logits = F.log_softmax(source_ft_logits,dim=-1)
                    
                target_zs_inputs = data[f'{target_zs_model.image_encoder.model_name}_features'].to(device=device)
                target_zs_inputs/=target_zs_inputs.norm(dim=-1,keepdim=True)    
                if args.setting =='dg':
                    target_zs_logits = target_zs_model.classification_head(target_zs_inputs)[:,:args.num_classes]
                else:
                    target_zs_logits = F.log_softmax(target_zs_model.classification_head(target_zs_inputs)[:,:args.num_classes],dim=-1)
                            
                if baseline_mode=='proxy':
                    logits = target_zs_logits + alpha*(source_ft_logits - source_zs_logits)
                elif baseline_mode=='base_change':
                    logits = source_ft_logits + alpha*(target_zs_logits - source_zs_logits)
                elif baseline_mode=='ensemble':
                    logits = alpha*source_ft_logits + (1-alpha)*target_zs_logits
                elif baseline_mode=='self_ensemble':
                    logits = alpha*source_ft_logits + (1-alpha)*source_zs_logits
                else:
                    raise NotImplementedError
                
            else:
                raise NotImplementedError

            if args.setting == 'base2novel':
                is_base = torch.tensor([label.item() in dataset.base_class_idx for label in data['labels']], device=device)
                base_pred = logits[is_base]
                base_pred[:,dataset.novel_class_idx] = -np.inf
                base_pred = base_pred.argmax(dim=1, keepdim=True).to(device)
                base_gt = y[is_base]
                base_correct += base_pred.eq(base_gt.view_as(base_pred)).sum().item()
                base_n += base_gt.size(0)
                
                # is_novel = torch.tensor([label.item() in dataset.novel_class_idx for label in data['labels']], device=device)
                novel_pred = logits[~is_base]
                novel_pred[:,dataset.base_class_idx] = -np.inf
                novel_pred = novel_pred.argmax(dim=1, keepdim=True).to(device)
                novel_gt = y[~is_base]
                novel_correct += novel_pred.eq(novel_gt.view_as(novel_pred)).sum().item()
                novel_n += novel_gt.size(0)                                
                n += y.size(0)
            else:
                
                if args.setting =='dg':
                    logits = logits[:,dataset.dg_class_idx]
                pred = logits.argmax(dim=1, keepdim=True).to(device)

                correct += pred.eq(y.view_as(pred)).sum().item()
                
                n += y.size(0)
        sharpness /= n
        if args.setting == 'base2novel':
            top1_base = base_correct / base_n
            top1_novel = novel_correct / novel_n
        else:
            top1 = correct / n
    
    if args.setting == 'base2novel':
        metrics = {'top1_base': top1_base, 'top1_novel': top1_novel}
        print(f'Done evaluating on {dataset_name}. Base Accuracy: {100*top1_base:.2f}%, Novel Accuracy: {100*top1_novel:.2f}% (sharpness : {sharpness:.4f})')
        
    else:
        metrics = {'top1': top1}
        print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}% (sharpness : {sharpness:.4f})')
    return metrics

def eval_baseline(model, dataset_name, args, is_base_finetuned=False):
    
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    dataset = get_dataset(
        dataset_name,
        # model.val_preprocess if not args.post_normalize else transforms.Compose(model.val_preprocess.transforms[:-1]),
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        args=args
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    with torch.no_grad():
        if args.setting == 'base2novel':
            top1_base, base_correct, base_n = 0., 0., 0.
            top1_novel, novel_correct, novel_n = 0., 0., 0.
            n=0
        else:
            top1, correct, n = 0., 0., 0.
        sharpness = 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            if args.use_precompute_features:
                x = None
                y = data['labels'].to('cuda:0')
                if is_base_finetuned:
                    all_logits = []
                    if args.use_ft_logits:
                        inputs = data[f'{model[0].image_encoder.model_name}_ft_logits']
                        for m,inp in zip(model,inputs):
                            logit = inp.to('cuda:0')
                            # if m.image_encoder.model.logit_scale is not None:
                            #     logit_scale = m.image_encoder.model.logit_scale.exp()
                            #     logit = logit*logit_scale
                            # if m.image_encoder.model.logit_bias is not None:
                            #     logit_bias = m.image_encoder.model.logit_bias
                            #     logit = logit + logit_bias
                            all_logits.append(logit[:,:args.num_classes])
                            
                    else:
                        inputs = data[f'{model[0].image_encoder.model_name}_ft_features']
                        for m,inp in zip(model,inputs):
                            inp = inp.to('cuda:0')
                            inp/=inp.norm(dim=-1,keepdim=True)
                            logits = m.classification_head(inp)[:,:args.num_classes]
                            all_logits.append(logits)
                            
                    if args.apply_tf_ensemble:
                        logits = torch.stack(all_logits)
                        prob = logits.softmax(dim=-1)
                        max_prob_per_model = prob.max(dim=-1,keepdim=True)[0]
                        weight_per_model = (max_prob_per_model*args.temperature).softmax(dim=0)
                        # weight_per_model = max_prob_per_model/max_prob_per_model.sum(dim=0,keepdim=True)
                        # log_prob = torch.log(prob)
                        logits = (weight_per_model*logits).sum(dim=0)
                        # sd_logits = (weight_per_model*logits).mean(dim=0)
                    else:
                        logits = torch.stack(all_logits).mean(dim=0)
                    sharpness += torch.logsumexp(logits,dim=-1).sum().item()
                else:
                    inputs = data[f'{model.image_encoder.model_name}_features'].to('cuda:0')
                    
                    inputs/=inputs.norm(dim=-1,keepdim=True)
                    logits = model.classification_head(inputs)[:,:args.num_classes]
                    sharpness += torch.logsumexp(logits,dim=-1).sum().item()
                
            else:
                x = data['images'].to(device)
                y = data['labels'].to(device)

                logits = model(x)

            if args.setting == 'base2novel':
                is_base = torch.tensor([label.item() in dataset.base_class_idx for label in data['labels']], device=device)
                base_pred = logits[is_base]
                base_pred[:,dataset.novel_class_idx] = -np.inf
                base_pred = base_pred.argmax(dim=1, keepdim=True).to(device)
                base_gt = y[is_base]
                base_correct += base_pred.eq(base_gt.view_as(base_pred)).sum().item()
                base_n += base_gt.size(0)
                
                # is_novel = torch.tensor([label.item() in dataset.novel_class_idx for label in data['labels']], device=device)
                novel_pred = logits[~is_base]
                novel_pred[:,dataset.base_class_idx] = -np.inf
                novel_pred = novel_pred.argmax(dim=1, keepdim=True).to(device)
                novel_gt = y[~is_base]
                novel_correct += novel_pred.eq(novel_gt.view_as(novel_pred)).sum().item()
                novel_n += novel_gt.size(0)                                
                n += y.size(0)
            else:
                if args.setting =='dg':
                    logits = logits[:,dataset.dg_class_idx]
                pred = logits.argmax(dim=1, keepdim=True).to(device)

                correct += pred.eq(y.view_as(pred)).sum().item()
                
                n += y.size(0)
        sharpness /= n
        if args.setting == 'base2novel':
            top1_base = base_correct / base_n
            top1_novel = novel_correct / novel_n
        else:
            top1 = correct / n
    
    if args.setting == 'base2novel':
        metrics = {'top1_base': top1_base, 'top1_novel': top1_novel, 'sharpness':sharpness}
        print(f'Done evaluating on {dataset_name}. Base Accuracy: {100*top1_base:.2f}%, Novel Accuracy: {100*top1_novel:.2f}% (sharpness : {sharpness:.4f})')
        
    else:
        metrics = {'top1': top1,'sharpness':sharpness}
        print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}% (sharpness : {sharpness:.4f})')
    
    return metrics

def eval_ensemble_multi_proxy(all_proxy_models, base_model, dataset_name, tf_ensemble=False, weights=None, args=None):
    base_model.eval()
    
    if weights is not None:
        weights = weights.to(device=args.device)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
    dataset = get_dataset(
        dataset_name,
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        args=args
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    with torch.no_grad():
        if args.setting == 'base2novel':
            top1_base, base_correct, base_n = 0., 0., 0.
            top1_novel, novel_correct, novel_n = 0., 0., 0.
            n=0
        else:
            top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            if args.use_precompute_features:
                zs_inputs = data[f'{base_model.image_encoder.model_name}_features'].to('cuda:0')
                x = None
                y = data['labels'].to('cuda:0')
                
                zs_inputs/=zs_inputs.norm(dim=-1,keepdim=True)
                logits = base_model.classification_head(zs_inputs)
                
                all_proxy_logits = []
                for proxy_model in all_proxy_models:
                    proxy_model.eval()
                    proxy_logits = proxy_model(x, logits)['logits'][:,:args.num_classes]
                    all_proxy_logits.append(proxy_logits)
                all_proxy_logits = torch.stack(all_proxy_logits)
                if tf_ensemble:
                    all_probs = all_proxy_logits.softmax(dim=-1)
                    max_prob_per_model = all_probs.max(dim=-1,keepdim=True)[0]
                    # Get the indices of top half most confident models
                    num_models = max_prob_per_model.size(0)
                    num_top_models = num_models // 2
                    top_indices = max_prob_per_model.squeeze(-1).topk(num_top_models, dim=0)[1]
                    # Create a mask for the selected models
                    mask = torch.zeros_like(max_prob_per_model)
                    mask[top_indices] = 1.0
                    # Apply mask and normalize weights
                    masked_probs = max_prob_per_model * mask
                    weight_per_model = masked_probs / (masked_probs.sum(dim=0, keepdim=True) + 1e-8)
                    logits = (weight_per_model * all_proxy_logits).sum(dim=0)
                    
                else:
                    if weights is None:
                        logits = all_proxy_logits.mean(dim=0)
                    else:
                        logits = (weights*all_proxy_logits).sum(dim=0)
                
            else:
                raise NotImplementedError

            if args.setting == 'base2novel':
                is_base = torch.tensor([label.item() in dataset.base_class_idx for label in data['labels']], device=device)
                base_pred = logits[is_base]
                base_pred[:,dataset.novel_class_idx] = -np.inf
                base_pred = base_pred.argmax(dim=1, keepdim=True).to(device)
                base_gt = y[is_base]
                base_correct += base_pred.eq(base_gt.view_as(base_pred)).sum().item()
                base_n += base_gt.size(0)
                
                # is_novel = torch.tensor([label.item() in dataset.novel_class_idx for label in data['labels']], device=device)
                novel_pred = logits[~is_base]
                novel_pred[:,dataset.base_class_idx] = -np.inf
                novel_pred = novel_pred.argmax(dim=1, keepdim=True).to(device)
                novel_gt = y[~is_base]
                novel_correct += novel_pred.eq(novel_gt.view_as(novel_pred)).sum().item()
                novel_n += novel_gt.size(0)                                
                n += y.size(0)
            else:
                if args.setting =='dg':
                    logits = logits[:,dataset.dg_class_idx]
                pred = logits.argmax(dim=1, keepdim=True).to(device)

                correct += pred.eq(y.view_as(pred)).sum().item()
            
                n += y.size(0)
            
        # top1 = correct / n
        if args.setting == 'base2novel':
            top1_base = base_correct / base_n
            top1_novel = novel_correct / novel_n
        else:
            top1 = correct / n
            
    if args.setting == 'base2novel':
        metrics = {'top1_base': top1_base, 'top1_novel': top1_novel}
        print(f'Ensemble Multi-Proxy Done evaluating on {dataset_name}. Base Accuracy: {100*top1_base:.2f}%, Novel Accuracy: {100*top1_novel:.2f}%')
    else:
        metrics = {'top1': top1}
        print(f'Ensemble Multi-Proxy Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
            
    # metrics = {'top1': top1}
    # print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics
                

def eval_ensemble_proxy(all_models, dataset_name, args, beta):
    source_proxy_model, target_proxy_model, model = all_models
    source_proxy_model.eval()
    target_proxy_model.eval()
    model.eval()
    
    dataset = get_dataset(
        dataset_name,
        target_proxy_model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        args=args
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    with torch.no_grad():
        if args.setting == 'base2novel':
            top1_base, base_correct, base_n = 0., 0., 0.
            top1_novel, novel_correct, novel_n = 0., 0., 0.
            n=0
        else:
            top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            if args.use_precompute_features:
                zs_inputs = data[f'{model.image_encoder.model_name}_features'].to('cuda:0')
                x = None
                y = data['labels'].to('cuda:0')
                
                zs_inputs/=zs_inputs.norm(dim=-1,keepdim=True)
                logits = model.classification_head(zs_inputs)
                
                # proxy_model.eval()
                source_logits = source_proxy_model(x, logits)['logits'][:,:args.num_classes]
                target_logits = target_proxy_model(x, logits)['logits'][:,:args.num_classes]
                if args.setting =='dg':
                    logits = (1-beta)*source_logits + beta * target_logits
                else:
                    # logits = (1-beta)*F.log_softmax(source_logits,dim=-1) + beta * F.log_softmax(target_logits,dim=-1)
                    logits = (1-beta)*source_logits + beta * target_logits
                
            else:
                x = data['images'].to(device)
                y = data['labels'].to(device)

                logits = model(x)
                source_logits = source_proxy_model(x, logits)['logits'][:,:args.num_classes]
                target_logits = target_proxy_model(x, logits)['logits'][:,:args.num_classes]
                if args.setting =='dg':
                    logits = (1-beta)*source_logits + beta * target_logits
                else:
                    # logits = (1-beta)*F.log_softmax(source_logits,dim=-1) + beta * F.log_softmax(target_logits,dim=-1)
                    logits = (1-beta)*source_logits + beta * target_logits

            if args.setting == 'base2novel':
                is_base = torch.tensor([label.item() in dataset.base_class_idx for label in data['labels']], device=device)
                base_pred = logits[is_base]
                base_pred[:,dataset.novel_class_idx] = -np.inf
                base_pred = base_pred.argmax(dim=1, keepdim=True).to(device)
                base_gt = y[is_base]
                base_correct += base_pred.eq(base_gt.view_as(base_pred)).sum().item()
                base_n += base_gt.size(0)
                
                # is_novel = torch.tensor([label.item() in dataset.novel_class_idx for label in data['labels']], device=device)
                novel_pred = logits[~is_base]
                novel_pred[:,dataset.base_class_idx] = -np.inf
                novel_pred = novel_pred.argmax(dim=1, keepdim=True).to(device)
                novel_gt = y[~is_base]
                novel_correct += novel_pred.eq(novel_gt.view_as(novel_pred)).sum().item()
                novel_n += novel_gt.size(0)                                
                n += y.size(0)
            else:
                if args.setting =='dg':
                    logits = logits[:,dataset.dg_class_idx]
                pred = logits.argmax(dim=1, keepdim=True).to(device)

                correct += pred.eq(y.view_as(pred)).sum().item()
            
                n += y.size(0)
            
        # top1 = correct / n
        if args.setting == 'base2novel':
            top1_base = base_correct / base_n
            top1_novel = novel_correct / novel_n
        else:
            top1 = correct / n
            
    if args.setting == 'base2novel':
        metrics = {'top1_base': top1_base, 'top1_novel': top1_novel}
        print(f'Done evaluating on {dataset_name}. Base Accuracy: {100*top1_base:.2f}%, Novel Accuracy: {100*top1_novel:.2f}%')
    else:
        metrics = {'top1': top1}
        print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
            
    # metrics = {'top1': top1}
    # print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics
    
def eval_single_dataset(image_encoder, dataset_name, args, save_dir=None, save_pred_dir=None, alpha=1.0, standardize=False):
    # Always expects [proxy_model, model] as input
    proxy_model, model = image_encoder
    
    model.eval()
    if save_pred_dir is not None:
        os.makedirs(save_pred_dir, exist_ok=True)
        pred_gt_dict = {}
        
    dataset = get_dataset(
        dataset_name,
        proxy_model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        args=args
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        if args.setting == 'base2novel':
            top1_base, base_correct, base_n = 0., 0., 0.
            top1_novel, novel_correct, novel_n = 0., 0., 0.
            n=0
        else:
            top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            if args.use_precompute_features:
                zs_inputs = data[f'{model.image_encoder.model_name}_features'].to('cuda:0')
                x = None
                y = data['labels'].to('cuda:0')
                
                zs_inputs = zs_inputs / zs_inputs.norm(dim=-1, keepdim=True)
                logits = model.classification_head(zs_inputs)
                
                proxy_model.eval()
                proxy_logits = proxy_model(x, logits)['logits'][:, :args.num_classes]
                if standardize:
                    logits = alpha * F.log_softmax(proxy_logits) + (1-alpha) * F.log_softmax(logits[:, :args.num_classes], dim=-1)
                else:
                    logits = alpha * proxy_logits + (1-alpha) * logits[:, :args.num_classes]
            else:
                x = data['images'].to(device)
                y = data['labels'].to(device)

                proxy_model.eval()
                logits = model(x)
                logits = proxy_model(x, logits)['logits'][:, :args.num_classes]

            if args.setting == 'base2novel':
                is_base = torch.tensor([label.item() in dataset.base_class_idx for label in data['labels']], device=device)
                base_pred = logits[is_base]
                base_pred[:,dataset.novel_class_idx] = -np.inf
                base_pred = base_pred.argmax(dim=1, keepdim=True).to(device)
                base_gt = y[is_base]
                base_correct += base_pred.eq(base_gt.view_as(base_pred)).sum().item()
                base_n += base_gt.size(0)
                
                # is_novel = torch.tensor([label.item() in dataset.novel_class_idx for label in data['labels']], device=device)
                novel_pred = logits[~is_base]
                novel_pred[:,dataset.base_class_idx] = -np.inf
                novel_pred = novel_pred.argmax(dim=1, keepdim=True).to(device)
                novel_gt = y[~is_base]
                novel_correct += novel_pred.eq(novel_gt.view_as(novel_pred)).sum().item()
                novel_n += novel_gt.size(0)                                
                n += y.size(0)
            else:
                if args.setting =='dg':
                    logits = logits[:,dataset.dg_class_idx]
                pred = logits.argmax(dim=1, keepdim=True).to(device)

                correct += pred.eq(y.view_as(pred)).sum().item()
                
                n += y.size(0)
                
            if save_pred_dir is not None:
                pred_gt_dict.update({f"{i}_pred":logits.cpu().tolist(),
                                     f"{i}_gt":y.cpu().tolist()})
        if args.setting == 'base2novel':
            top1_base = base_correct / base_n
            top1_novel = novel_correct / novel_n
        else:
            top1 = correct / n
    if save_pred_dir is not None:
        filename = os.path.join(save_pred_dir, f'pred_gt.json')
        with open(filename, 'w') as f:
            json.dump(pred_gt_dict, f)
            
    if args.setting == 'base2novel':
        metrics = {'top1_base': top1_base, 'top1_novel': top1_novel}
        print(f'Done evaluating on {dataset_name}. Base Accuracy: {100*top1_base:.2f}%, Novel Accuracy: {100*top1_novel:.2f}%')
        
    else:
        metrics = {'top1': top1}
        print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics

def eval_delta(source_model, target_model, proxy_model, dataset_name, args, save_dir=None, save_pred_dir=None, alpha=1.0):
    source_model.eval()
    target_model.eval()
    proxy_model.eval()
        
    dataset = get_dataset(
        dataset_name,
        proxy_model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        args=args
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        if args.setting == 'base2novel':
            top1_base, base_correct, base_n = 0., 0., 0.
            top1_novel, novel_correct, novel_n = 0., 0., 0.
            n=0
        else:
            top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            if args.use_precompute_features:
                source_zs_inputs = data[f'{source_model.image_encoder.model_name}_features'].to('cuda:0')
                target_zs_inputs = data[f'{target_model.image_encoder.model_name}_features'].to('cuda:0')
                x = None
                y = data['labels'].to('cuda:0')
                
                source_zs_inputs/=source_zs_inputs.norm(dim=-1,keepdim=True)
                target_zs_inputs/=target_zs_inputs.norm(dim=-1,keepdim=True)
                source_logits = source_model.classification_head(source_zs_inputs)
                target_logits = target_model.classification_head(target_zs_inputs)
                
                source_delta = proxy_model(x, source_logits)['logits'][:,:args.num_classes] - source_logits[:,:args.num_classes]
                logits = target_logits[:,:args.num_classes] + source_delta*alpha
                
            else:
                x = data['images'].to(device)
                y = data['labels'].to(device)

                
                proxy_model.eval()
                # base_model.eval()
                source_logits = source_model(x)
                source_delta = proxy_model(x, source_logits)['logits'][:,:args.num_classes] - source_logits[:,:args.num_classes]
                logits = target_model(x)[:,:args.num_classes] + source_delta
                

            if args.setting == 'base2novel':
                is_base = torch.tensor([label.item() in dataset.base_class_idx for label in data['labels']], device=device)
                base_pred = logits[is_base]
                base_pred[:,dataset.novel_class_idx] = -np.inf
                base_pred = base_pred.argmax(dim=1, keepdim=True).to(device)
                base_gt = y[is_base]
                base_correct += base_pred.eq(base_gt.view_as(base_pred)).sum().item()
                base_n += base_gt.size(0)
                
                # is_novel = torch.tensor([label.item() in dataset.novel_class_idx for label in data['labels']], device=device)
                novel_pred = logits[~is_base]
                novel_pred[:,dataset.base_class_idx] = -np.inf
                novel_pred = novel_pred.argmax(dim=1, keepdim=True).to(device)
                novel_gt = y[~is_base]
                novel_correct += novel_pred.eq(novel_gt.view_as(novel_pred)).sum().item()
                novel_n += novel_gt.size(0)                                
                n += y.size(0)
            else:
                if args.setting =='dg':
                    logits = logits[:,dataset.dg_class_idx]
                pred = logits.argmax(dim=1, keepdim=True).to(device)

                correct += pred.eq(y.view_as(pred)).sum().item()
                
                n += y.size(0)

        if args.setting == 'base2novel':
            top1_base = base_correct / base_n
            top1_novel = novel_correct / novel_n
        else:
            top1 = correct / n

    if args.setting == 'base2novel':
        metrics = {'top1_base': top1_base, 'top1_novel': top1_novel}
        print(f'Done evaluating on {dataset_name}. Base Accuracy: {100*top1_base:.2f}%, Novel Accuracy: {100*top1_novel:.2f}%')
        
    else:
        metrics = {'top1': top1}
        print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics

def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info

def evaluate_with_prelogit(logit_dir,file_name,dataset_name,proxy_model, args, logger=None):
    proxy_model.eval()
    file_path = os.path.join(logit_dir, file_name)
    with open(file_path, 'r') as f:
        logit_dict = json.load(f)
    logger.info(f"Loaded logit dict from {file_path}")
    
    dataset = get_dataset(
        dataset_name,
        proxy_model.val_preprocess,
        location=args.data_location,
        batch_size=128
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        tot_pre_sharpness, tot_cur_sharpness = 0., 0.
        pre_entropy, cur_entropy = 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            assert (data['labels'] == torch.tensor(logit_dict[f"{i}_gt"])).all(), "GT labels do not match"
            
            pre_logits = torch.tensor(logit_dict[f"{i}_pred"]).to(device)
            # pred = pre_logits.argmax(dim=1, keepdim=True).to(device)
            logits = proxy_model(x,pre_logits)['logits']
            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            tot_pre_sharpness += torch.logsumexp(pre_logits,dim=-1).sum().item()
            tot_cur_sharpness += torch.logsumexp(logits,dim=-1).sum().item()
            pre_entropy += softmax_entropy(pre_logits).sum().item()
            cur_entropy += softmax_entropy(logits).sum().item()
            
            n += y.size(0)
        top1 = correct / n
        tot_pre_sharpness /= n
        tot_cur_sharpness /= n
        pre_entropy/=n
        cur_entropy/=n
        
    metrics = {'top1': top1, 'pre_sharpness': tot_pre_sharpness, 'cur_sharpness': tot_cur_sharpness,
               'pre_entropy': pre_entropy, 'cur_entropy': cur_entropy}
    logger.info(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    logger.info(f"Pre-Sharpness: {tot_pre_sharpness:.4f}, Cur-Sharpness: {tot_cur_sharpness:.4f}")
    logger.info(f"Pre-Entropy: {pre_entropy:.4f}, Cur-Entropy: {cur_entropy:.4f}")
    return metrics

def eval_correlation(source_model, source_ft_model, target_model, proxy_model, dataset_name, args, logger=None):
    source_model.eval()
    target_model.eval()
    proxy_model.eval()
    
    dataset = get_dataset(
        dataset_name,
        proxy_model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        args=args
    )
    
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        all_delta_corr = []
        all_delta_kl_div = []
        all_delta_cos_source = []
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            
            if args.use_precompute_features:
                source_zs_inputs = data[f'{source_model.image_encoder.model_name}_features'].to(device=device)
                source_zs_inputs/=source_zs_inputs.norm(dim=-1,keepdim=True)
                source_zs_logits = source_model.classification_head(source_zs_inputs)[:,:args.num_classes]
                
                all_source_ft_logits = []
                if args.use_ft_logits:
                    source_ft_inputs = data[f'{source_model.image_encoder.model_name}_ft_logits']
                    for ft_mod, ft_inp in zip(source_ft_model, source_ft_inputs):
                        ft_inp = ft_inp.to(device=device)
                    
                        all_source_ft_logits.append(ft_inp[:,:args.num_classes])
                    # source_ft_logits = torch.stack(all_source_ft_logits).mean(dim=0)[:,:args.num_classes]
                else:
                    source_ft_inputs = data[f'{source_model.image_encoder.model_name}_ft_features']
                    for ft_mod, ft_inp in zip(source_ft_model, source_ft_inputs):
                        ft_mod.eval()
                        ft_inp = ft_inp.to(device=device)
                        ft_inp/=ft_inp.norm(dim=-1,keepdim=True)
                        source_ft_logits = ft_mod.classification_head(ft_inp)
                        all_source_ft_logits.append(source_ft_logits[:,:args.num_classes])
                
                if args.apply_tf_ensemble:
                    source_ft_logits = torch.stack(all_source_ft_logits)
                    prob = source_ft_logits.softmax(dim=-1)
                    max_prob_per_model = prob.max(dim=-1,keepdim=True)[0]
                    weight_per_model = max_prob_per_model.softmax(dim=0)
                    source_ft_logits = (weight_per_model*source_ft_logits).sum(dim=0)
                else:
                    source_ft_logits = torch.stack(all_source_ft_logits).mean(dim=0)[:,:args.num_classes]
                    
                delta_source_orig = source_ft_logits-source_zs_logits
                
                target_zs_inputs = data[f'{target_model.image_encoder.model_name}_features'].to(device=device)
                target_zs_inputs/=target_zs_inputs.norm(dim=-1,keepdim=True)
                target_zs_logits = target_model.classification_head(target_zs_inputs)
                target_proxy_logits = proxy_model(None,target_zs_logits)['logits'][:,:args.num_classes]
                delta_target_orig = target_proxy_logits-target_zs_logits[:,:args.num_classes]
                
                # calculate correlaction, kl distance, and cosine similarity
                # corr_source = torch.corrcoef(torch.cat([delta_source_orig,delta_target_orig],dim=1))[0,1]
                
                centered_delta_source_orig = delta_source_orig-delta_source_orig.mean(dim=-1,keepdim=True)
                centered_delta_source_orig/=centered_delta_source_orig.norm(dim=-1,keepdim=True)
                centered_delta_target_orig = delta_target_orig-delta_target_orig.mean(dim=-1,keepdim=True)
                centered_delta_target_orig/=centered_delta_target_orig.norm(dim=-1,keepdim=True)
                
                delta_corr = (centered_delta_source_orig*centered_delta_target_orig).sum(dim=-1)
                delta_kl_div = F.kl_div(F.log_softmax(delta_source_orig,dim=-1),F.softmax(delta_target_orig,dim=-1),reduction='none').sum(dim=-1)
                delta_cos_source = F.cosine_similarity(delta_source_orig,delta_target_orig,dim=-1)
                
                all_delta_corr.append(delta_corr)
                all_delta_kl_div.append(delta_kl_div)
                all_delta_cos_source.append(delta_cos_source)
                
            else:
                raise NotImplementedError
                
        all_delta_corr = torch.cat(all_delta_corr,dim=0)
        all_delta_kl_div = torch.cat(all_delta_kl_div,dim=0)
        all_delta_cos_source = torch.cat(all_delta_cos_source,dim=0)
        
        # corr_source = torch.corrcoef(torch.cat([all_delta_source_orig,all_delta_target_orig],dim=1))[0,1]
        corr_pearson = all_delta_corr.mean()
        corr_kl = all_delta_kl_div.mean()
        corr_cos = all_delta_cos_source.mean()
        
        metrics = {'pearson_corr': corr_pearson, 'kl_div_corr': corr_kl, 'cos_corr': corr_cos}
        # logger.info(f"Correlation: {corr_mean:.4f}, KL Divergence: {kl_div_mean:.4f}, Cosine Similarity: {cos_mean:.4f}")
        return metrics
            
