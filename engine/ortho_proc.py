import os
import json
import tqdm
import gc

import torch
import numpy as np
import sys
from scipy.linalg import logm, expm

from utils import utils
from datasets.common import get_dataloader, maybe_dictionarize
from models.heads import get_classification_head
from models.modeling import ImageClassifier

from datasets.registry import get_dataset

def block_matmul(A,B,block_size=1024,device='cpu'):
    result = torch.zeros(A.shape[1], B.shape[1], device=device)
    for i in tqdm.tqdm(range(0, A.shape[0], block_size)):
        a_block = A[i:i+block_size].to(device=device)
        b_block = B[i:i+block_size].to(device=  device)
        result += a_block.T @ b_block
    return result

def run_models(source_models, target_models, dataset, args, w_old=None, ):
    
    source_zs_model,source_proxy_model = source_models
    target_zs_model,target_proxy_model = target_models
    
    source_zs_model.eval()
    source_proxy_model.eval()
    target_zs_model.eval()
    target_proxy_model.eval()
    
    source_proxy_model.network[0].transfer_procrutes_mode = True
    target_proxy_model.network[0].transfer_procrutes_mode = True
    
    source_proxy_model.network[0].pre_compute_stats = args.source_stats
    target_proxy_model.network[0].pre_compute_stats = args.target_stats    
    
    data_loader = get_dataloader(
            dataset,is_train=True, args=args, image_encoder=None, is_transfer=True)
    device = args.device

    # logit_list={'target':[],'source':[]}
    source_lists , target_lists = [],[]
    # count = 0
    with torch.no_grad():
        
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            data = maybe_dictionarize(data,procrutes_mode=True)
            
            if args.use_precompute_features:
                x = None
                source_feats = data[f'{source_zs_model.image_encoder.model_name}_features'].to(device)
                target_feats = data[f'{target_zs_model.image_encoder.model_name}_features'].to(device)
                source_feats /= source_feats.norm(dim=-1, keepdim=True)
                target_feats /= target_feats.norm(dim=-1, keepdim=True)
                source_logit = source_zs_model.classification_head(source_feats)
                target_logit = target_zs_model.classification_head(target_feats)
            else:
                x = data['images'].to(device)
                # y = data['labels'].to(device)

                source_logit = source_zs_model(x)
                target_logit = target_zs_model(x)
            
            source_preprocess_logit = source_proxy_model(x,source_logit)['logits']
            target_preprocess_logit = target_proxy_model(x,target_logit)['logits']
            
            if w_old is not None:
                source_lists.append((source_preprocess_logit@w_old.T).cpu())
                target_lists.append((target_preprocess_logit@w_old.T).cpu())
            else:
                source_lists.append(source_preprocess_logit.cpu())
                target_lists.append(target_preprocess_logit.cpu())
    source_lists = torch.cat(source_lists,dim=0)
    target_lists = torch.cat(target_lists,dim=0)
    
    source_proxy_model.network[0].transfer_procrutes_mode = False
    target_proxy_model.network[0].transfer_procrutes_mode = False
    
    return target_lists, source_lists

def run_models_ver2(source_models, target_models, dataset, args, w_old=None, ):
    
    source_zs_model,source_proxy_model = source_models
    target_zs_model,target_proxy_model = target_models
    
    source_zs_model.eval()
    source_proxy_model.eval()
    target_zs_model.eval()
    target_proxy_model.eval()
    
    source_proxy_model.network[0].transfer_procrutes_mode = True
    target_proxy_model.network[0].transfer_procrutes_mode = True
    
    source_proxy_model.network[0].pre_compute_stats = args.source_stats
    target_proxy_model.network[0].pre_compute_stats = args.target_stats    
    
    data_loader = get_dataloader(
            dataset,is_train=True, args=args, image_encoder=None, is_transfer=True)
    device = args.device

    # logit_list={'target':[],'source':[]}
    if 'both_feats' in args.extract_latent_place:
        target_feat_list={'in':[],'out':[]}
        source_feat_list={'in':[],'out':[]}
    else:
        
        target_feat_list=[]
        source_feat_list=[]
    # count = 0
    with torch.no_grad():
        # data_loader = get_dataloader(
        #     dataset,is_train=True, args=args, image_encoder=None, is_transfer=True)
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            data = maybe_dictionarize(data,procrutes_mode=True)
            
            if args.use_precompute_features:
                x = None
                source_feats = data[f'{source_zs_model.image_encoder.model_name}_features'].to(device)
                target_feats = data[f'{target_zs_model.image_encoder.model_name}_features'].to(device)
                source_feats /= source_feats.norm(dim=-1, keepdim=True)
                target_feats /= target_feats.norm(dim=-1, keepdim=True)
                source_logit = source_zs_model.classification_head(source_feats)
                target_logit = target_zs_model.classification_head(target_feats)
            else:
                x = data['images'].to(device)
                target_logit = target_zs_model(x)
                source_logit = source_zs_model(x)
    
            if w_old is not None:
                source_feats = source_proxy_model(x,source_logit)['logits']@w_old.T
                target_feats = target_proxy_model(x,target_logit)['logits']@w_old.T
            else:
                source_feats = source_proxy_model(x,source_logit)['logits']
                target_feats = target_proxy_model(x,target_logit)['logits']
                
            if 'both_feats' in args.extract_latent_place:
                target_feat_list['in'].append(target_feats[0].cpu())
                target_feat_list['out'].append(target_feats[1].cpu())
                source_feat_list['in'].append(source_feats[0].cpu())
                source_feat_list['out'].append(source_feats[1].cpu())
            else:
                source_feat_list.append(source_feats.cpu())
                target_feat_list.append(target_feats.cpu())

            if args.load_transferred_model:
                break
            
        if 'both_feats' in args.extract_latent_place:
            target_feat_list['in'] = torch.cat(target_feat_list['in'],dim=0)
            target_feat_list['out'] = torch.cat(target_feat_list['out'],dim=0)
            source_feat_list['in'] = torch.cat(source_feat_list['in'],dim=0)
            source_feat_list['out'] = torch.cat(source_feat_list['out'],dim=0)
        else:
            target_feat_list = torch.cat(target_feat_list,dim=0)
            source_feat_list = torch.cat(source_feat_list,dim=0)
        
    #     block_size = 1024  # You can adjust the block size based on your GPU memory
    #     result = torch.zeros(target_feat_list.shape[1], source_feat_list.shape[1], device=device)

    #     for i in tqdm.tqdm(range(0, target_feat_list.shape[0], block_size)):
    #         target_block = target_feat_list[i:i+block_size].to(device)
    #         source_block = source_feat_list[i:i+block_size].to(device)
    #         result += target_block.T @ source_block
    # del target_feat_list, source_feat_list
    # gc.collect()
    # torch.cuda.empty_cache()
        # result = result.cpu()
    source_proxy_model.network[0].transfer_procrutes_mode = False
    target_proxy_model.network[0].transfer_procrutes_mode = False
    
    return target_feat_list, source_feat_list
    # return torch.cat(logit_list['source'],dim=0),torch.cat(logit_list['target'],dim=0)

def run_models_ensemble(source_models, target_models, dataset, args, w_old=None, ):
    
    source_zs_model,source_proxy_model = source_models
    target_zs_model,target_proxy_model = target_models
    
    source_zs_model.eval()
    # source_proxy_model.eval()
    target_zs_model.eval()
    # target_proxy_model.eval()
    
    for m in source_proxy_model.transmiters:
        m.network[0].transfer_procrutes_mode = True
    for m in target_proxy_model.transmiters:
        m.network[0].transfer_procrutes_mode = True
    for m in source_proxy_model.transmiters:
        m.network[0].pre_compute_stats = args.source_stats
    for m in target_proxy_model.transmiters:
        m.network[0].pre_compute_stats = args.target_stats    
    
    data_loader = get_dataloader(
            dataset,is_train=True, args=args, image_encoder=None, is_transfer=True)
    device = args.device

    # logit_list={'target':[],'source':[]}
    if 'both_feats' in args.extract_latent_place:
        raise NotImplementedError
        target_feat_list={'in':[],'out':[]}
        source_feat_list={'in':[],'out':[]}
    else:
        
        source_feat_list=[[]]*len(args.ft_strategy) if args.seperate_ensemble else []
        target_feat_list=[[]]*len(args.ft_strategy) if args.seperate_ensemble else []
    # count = 0
    with torch.no_grad():
        # data_loader = get_dataloader(
        #     dataset,is_train=True, args=args, image_encoder=None, is_transfer=True)
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            data = maybe_dictionarize(data,procrutes_mode=True)
            
            if args.use_precompute_features:
                x = None
                source_feats = data[f'{source_zs_model.image_encoder.model_name}_features'].to(device)
                target_feats = data[f'{target_zs_model.image_encoder.model_name}_features'].to(device)
                source_feats /= source_feats.norm(dim=-1, keepdim=True)
                target_feats /= target_feats.norm(dim=-1, keepdim=True)
                source_logit = source_zs_model.classification_head(source_feats)
                target_logit = target_zs_model.classification_head(target_feats)
            else:
                x = data['images'].to(device)
                target_logit = target_zs_model(x)
                source_logit = source_zs_model(x)
    
            if w_old is not None:
                source_feats = source_proxy_model(x,source_logit, return_as_list=True)['logits']
                # source_feats = [s@w_old.T for s in source_feats]
                for ii,s in enumerate(source_feats):
                    ss = s@w_old[ii].T
                    source_feat_list[ii].append(ss.cpu())
                target_feats = target_proxy_model(x,target_logit, return_as_list=True)['logits']
                for ii,t in enumerate(target_feats):
                    tt = t@w_old[ii].T
                    target_feat_list[ii].append(tt.cpu())
            else:
                source_feats = source_proxy_model(x,source_logit,return_as_list=True)['logits']
                target_feats = target_proxy_model(x,target_logit,return_as_list=True)['logits']
                
            if 'both_feats' in args.extract_latent_place:
                target_feat_list['in'].append(target_feats[0].cpu())
                target_feat_list['out'].append(target_feats[1].cpu())
                source_feat_list['in'].append(source_feats[0].cpu())
                source_feat_list['out'].append(source_feats[1].cpu())
            # else:
            #     source_feat_list.append(source_feats.cpu())
            #     target_feat_list.append(target_feats.cpu())
        
        if 'both_feats' in args.extract_latent_place:
            target_feat_list['in'] = torch.cat(target_feat_list['in'],dim=0)
            target_feat_list['out'] = torch.cat(target_feat_list['out'],dim=0)
            source_feat_list['in'] = torch.cat(source_feat_list['in'],dim=0)
            source_feat_list['out'] = torch.cat(source_feat_list['out'],dim=0)
        else:
            target_feat_list = [torch.cat(tfl,dim=0) for tfl in target_feat_list]
            source_feat_list = [torch.cat(sfl,dim=0) for sfl in source_feat_list]

    for m in source_proxy_model.transmiters:
        m.network[0].transfer_procrutes_mode = False
    for m in target_proxy_model.transmiters:
        m.network[0].transfer_procrutes_mode = False
    
    return target_feat_list, source_feat_list

def norm_st(tensor,args):
    out_dict = {}
    if args.procrutes_norm_strategy == 'standard':
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        tensor = (tensor - mean) / std
        out_dict.update({'mean': mean, 'std': std, 'tensor': tensor})
    elif args.procrutes_norm_strategy == 'center':
        mean = tensor.mean(dim=0, keepdim=True)
        tensor = tensor - mean
        out_dict.update({'mean': mean, 'tensor': tensor})
    elif args.procrutes_norm_strategy == 'norm':
        tensor = tensor / tensor.norm(dim=-1, keepdim=True)
        out_dict.update({'tensor': tensor})
    elif args.procrutes_norm_strategy == 'none':
        tensor = tensor
        out_dict.update({'tensor': tensor})
    return out_dict

def orthogonal_procrutes_ensemble(z_new,z_old,w_old,args=None,w_out=None):
    ## z_new : target logits with shape (N,C)
    ## z_old : source logits with shape (N,C)
    ## w_old : source projector with shape (D,C)
    if args.pseudo_align:
        out_dict = {}
        B = z_old @ w_old.T
        A = z_new @ w_old.T
        A_pseudo  = torch.linalg.inv(A.T@A)@A.T
        sol = A_pseudo@B
        
        out_dict['W_in'] = sol.T@w_old
        out_dict['W_out'] = sol.T@w_old
        out_dict['in_bias'] = None
        out_dict['out_bias'] = None
        
        return out_dict
    
    if args.direct_procrutes:
        
        assert not args.latent_procrutes, 'latent_procrutes should be False'
        assert args.extract_latent_place == 'none', 'extract_latent_place should be none'
        
        out_dict={}
        R = block_matmul(z_new,z_old,device=args.device)
        # R=z_new.T@z_old
        
        R = R + args.reg_procrutes_coef * torch.eye(R.shape[0],device=R.device,dtype=R.dtype)
        U, S, V = torch.svd(R)
        if args.rotation:
            sv = torch.eye(S.shape[0], device=S.device)
            sv[-1,-1] = torch.det(U@V.T).sign()
            sol = U@sv@V.T
        else:
            # if args.truncate_ratio:
                # U = U[:,:args.num_classes]
                # V = V[:,:args.num_classes]
                # S[S<S.max()*1e-10]=0.
                # red_d = (S>S.max()*1e-8).sum()
            # red_d = args.num_classes
            red_d = (S>S.max()*args.truncate_ratio).sum()
            U=U[:,:red_d]
            V=V[:,:red_d]
            sol = U@V.T
        out_dict['W_in'] = w_old@sol.T
        out_dict['W_out'] = w_old@sol.T
        out_dict['in_bias'] = None
        out_dict['out_bias'] = None
        out_dict['sol'] = sol
        return out_dict
    
    if args.latent_procrutes:
        dict_list = []
        for i,(zn, zo, wo) in enumerate(zip(z_new,z_old,w_old)):
            out_dict={}
            # B = z_old @ w_old.T
            # A = z_new @ w_old.T
            # B = z_old
            # A = z_new
            
            R = block_matmul(zn,zo,device=args.device)
            # R=z_new.T@z_old
            
            R = R + args.reg_procrutes_coef * torch.eye(R.shape[0],device=R.device,dtype=R.dtype)
            U, S, V = torch.svd(R)
            if args.rotation:
                sv = torch.eye(S.shape[0], device=S.device)
                sv[-1,-1] = torch.det(U@V.T).sign()
                sol = U@sv@V.T
            else:
                # if args.truncate_ratio:
                    # U = U[:,:args.num_classes]
                    # V = V[:,:args.num_classes]
                    # S[S<S.max()*1e-10]=0.
                    # red_d = (S>S.max()*1e-8).sum()
                # red_d = args.num_classes
                red_d = (S>S.max()*args.truncate_ratio).sum().item()

                # S[S>0]=1.
                U=U[:,:red_d]
                V=V[:,:red_d]
                sol = U@V.T
            if args.extract_latent_place!='none':
                out_dict['latent'] = sol
                out_dict['W_in'] = None
                out_dict['W_out'] = None
                out_dict['in_bias'] = None
                out_dict['out_bias'] = None
            else:
                out_dict['W_in'] = sol.T@wo[i]
                out_dict['W_out'] = sol.T@wo[i] if w_out is None else sol.T@w_out[i].T
                out_dict['in_bias'] = None
                out_dict['out_bias'] = None
            dict_list.append(out_dict)
        # out_dict['sol'] = sol
        return dict_list
        
    if args.only_task_class_transfer:
        assert args.class_padding_strategy != 'none', 'class_padding_strategy should be not none'
        w_old_cat = torch.block_diag(w_old,torch.eye(z_old.shape[1]-args.num_classes,device=w_old.device,dtype=w_old.dtype))
        # B = z_old @ w_old_cat.T - z_new[:, args.num_classes:] @ w_old_cat[:, args.num_classes:].T
        # B = z_old @ w_old_cat.T
        # A = z_new
        out_dict = {}
        B_dict = norm_st(z_old,args)
        B = B_dict['tensor']@w_old_cat.T
        
        B_task, B_neg = B.split([args.num_classes, B.shape[1]-args.num_classes], dim=1)
        
        A_dict = norm_st(z_new,args)
        A = A_dict['tensor']
        A_task, A_neg = A.split([args.num_classes, A.shape[1]-args.num_classes], dim=1)
        
        
        R_task = B_task.T @ A_task
        R_neg = B_neg.T @ A_neg
        
        U_task, S_task, V_task = torch.svd(R_task)
        U_neg, S_neg, V_neg = torch.svd(R_neg)
        
        if args.rotation:
            sv_task = torch.eye(S_task.shape[0], device=S_task.device)
            sv_neg = torch.eye(S_neg.shape[0], device=S_neg.device)
            sv_task[-1,-1] = torch.det(U_task@V_task.T).sign()
            sv_neg[-1,-1] = torch.det(U_neg@V_neg.T).sign()
            sol_task = U_task@sv_task@V_task.T
            sol_neg = U_neg@sv_neg@V_neg.T
        
        else:
            sol_task = U_task@V_task.T
            sol_neg = U_neg@V_neg.T
        out_dict['W_in'] = torch.block_diag(sol_task,sol_neg)
        out_dict['W_out'] = torch.block_diag(sol_task,sol_neg)
        out_dict['in_bias'] = None
        out_dict['out_bias'] = None
        
        return out_dict
    else:
        bias=None
        sol_dict={}
        B = z_old @ w_old.T
        A = z_new
        
        B_dict = norm_st(z_old,args)
        A_dict = norm_st(A,args)
        B = B_dict['tensor']@ w_old.T
        A = A_dict['tensor']
        R = B.T @ A
        U, S, V = torch.svd(R)
        
        # S[S<S.max()*1e-10]=0.
        # S[S>0]=1.
        # sol = U@torch.diag(S)@V.T
        
        # sol = U @ V.T
        if args.rotation:
            sv = torch.eye(S.shape[0], device=S.device)
            sv[-1,-1] = torch.det(U@V.T).sign()
            sol = U@sv@V.T
        else:
            sol = U@V.T
        if args.procrutes_norm_strategy == 'standard':
            A_diag = torch.diag(A_dict['std'].squeeze(0))
            B_diag = torch.diag(B_dict['std'].squeeze(0))
            # sol = sol * B_dict['std'].squeeze(0).unsqueeze(-1) / A_dict['std']
            sol_dict['W_in'] = B_diag @ sol @ A_diag.inverse()
            in_bias = B_dict['mean'] - A_dict['mean']@sol_dict['W_in'].transpose(0,1)
            sol_dict['in_bias'] = in_bias.squeeze(0)
            
            sol_dict['W_out'] = B_diag.inverse()@sol@A_diag
            sol_dict['out_bias'] = -sol_dict['in_bias']
        elif args.procrutes_norm_strategy == 'center':
            bias = B_dict['mean'] - A_dict['mean']@sol.transpose(0,1)
            sol_dict['W_in'] = sol
            sol_dict['in_bias'] = bias.squeeze(0)
            sol_dict['W_out'] = sol
            sol_dict['out_bias'] = -sol_dict['in_bias']
        else:
            sol_dict['W_in']=sol
            sol_dict['in_bias']=None
            sol_dict['W_out']=sol
            
            sol_dict['out_bias']=None
        return sol_dict

def orthogonal_procrutes(z_new,z_old,w_old,args=None,w_out=None):
    ## z_new : target logits with shape (N,C)
    ## z_old : source logits with shape (N,C)
    ## w_old : source projector with shape (D,C)
    if args.pseudo_align:
        out_dict = {}
        B = z_old @ w_old.T
        A = z_new @ w_old.T
        A_pseudo  = torch.linalg.inv(A.T@A)@A.T
        sol = A_pseudo@B
        
        out_dict['W_in'] = sol.T@w_old
        out_dict['W_out'] = sol.T@w_old
        out_dict['in_bias'] = None
        out_dict['out_bias'] = None
        
        return out_dict
    
    if args.direct_procrutes:
        
        assert not args.latent_procrutes, 'latent_procrutes should be False'
        assert args.extract_latent_place == 'none', 'extract_latent_place should be none'
        
        out_dict={}
        R = block_matmul(z_new,z_old,device=args.device)
        # R=z_new.T@z_old
        
        R = R + args.reg_procrutes_coef * torch.eye(R.shape[0],device=R.device,dtype=R.dtype)
        U, S, V = torch.svd(R)
        if args.rotation:
            sv = torch.eye(S.shape[0], device=S.device)
            sv[-1,-1] = torch.det(U@V.T).sign()
            sol = U@sv@V.T
        else:
            # if args.truncate_ratio:
                # U = U[:,:args.num_classes]
                # V = V[:,:args.num_classes]
                # S[S<S.max()*1e-10]=0.
                # red_d = (S>S.max()*1e-8).sum()
            # red_d = args.num_classes
            red_d = (S>S.max()*args.truncate_ratio).sum()
            U=U[:,:red_d]
            V=V[:,:red_d]
            sol = U@V.T
        out_dict['W_in'] = w_old@sol.T
        out_dict['W_out'] = w_old@sol.T
        out_dict['in_bias'] = None
        out_dict['out_bias'] = None
        out_dict['sol'] = sol
        return out_dict
    
    if args.latent_procrutes:
        out_dict={}
        # B = z_old @ w_old.T
        # A = z_new @ w_old.T
        # B = z_old
        # A = z_new
        
        R = block_matmul(z_new,z_old,device=args.device)
        # R=z_new.T@z_old
        
        R = R + args.reg_procrutes_coef * torch.eye(R.shape[0],device=R.device,dtype=R.dtype)
        U, S, V = torch.svd(R)
        if args.rotation:
            sv = torch.eye(S.shape[0], device=S.device)
            sv[-1,-1] = torch.det(U@V.T).sign()
            sol = U@sv@V.T
        else:
            # if args.truncate_ratio:
                # U = U[:,:args.num_classes]
                # V = V[:,:args.num_classes]
                # S[S<S.max()*1e-10]=0.
                # red_d = (S>S.max()*1e-8).sum()
            # red_d = args.num_classes
            red_d = (S>S.max()*args.truncate_ratio).sum().item()

            # S[S>0]=1.
            U=U[:,:red_d]
            V=V[:,:red_d]
            sol = U@V.T
        if args.extract_latent_place!='none':
            out_dict['latent'] = sol
            out_dict['W_in'] = None
            out_dict['W_out'] = None
            out_dict['in_bias'] = None
            out_dict['out_bias'] = None
        else:
            out_dict['W_in'] = sol.T@w_old
            out_dict['W_out'] = sol.T@w_old if w_out is None else sol.T@w_out.T
            out_dict['in_bias'] = None
            out_dict['out_bias'] = None
        
        # out_dict['sol'] = sol
        return out_dict
        
    if args.only_task_class_transfer:
        assert args.class_padding_strategy != 'none', 'class_padding_strategy should be not none'
        w_old_cat = torch.block_diag(w_old,torch.eye(z_old.shape[1]-args.num_classes,device=w_old.device,dtype=w_old.dtype))
        # B = z_old @ w_old_cat.T - z_new[:, args.num_classes:] @ w_old_cat[:, args.num_classes:].T
        # B = z_old @ w_old_cat.T
        # A = z_new
        out_dict = {}
        B_dict = norm_st(z_old,args)
        B = B_dict['tensor']@w_old_cat.T
        
        B_task, B_neg = B.split([args.num_classes, B.shape[1]-args.num_classes], dim=1)
        
        A_dict = norm_st(z_new,args)
        A = A_dict['tensor']
        A_task, A_neg = A.split([args.num_classes, A.shape[1]-args.num_classes], dim=1)
        
        
        R_task = B_task.T @ A_task
        R_neg = B_neg.T @ A_neg
        
        U_task, S_task, V_task = torch.svd(R_task)
        U_neg, S_neg, V_neg = torch.svd(R_neg)
        
        if args.rotation:
            sv_task = torch.eye(S_task.shape[0], device=S_task.device)
            sv_neg = torch.eye(S_neg.shape[0], device=S_neg.device)
            sv_task[-1,-1] = torch.det(U_task@V_task.T).sign()
            sv_neg[-1,-1] = torch.det(U_neg@V_neg.T).sign()
            sol_task = U_task@sv_task@V_task.T
            sol_neg = U_neg@sv_neg@V_neg.T
        
        else:
            sol_task = U_task@V_task.T
            sol_neg = U_neg@V_neg.T
        out_dict['W_in'] = torch.block_diag(sol_task,sol_neg)
        out_dict['W_out'] = torch.block_diag(sol_task,sol_neg)
        out_dict['in_bias'] = None
        out_dict['out_bias'] = None
        
        return out_dict
    else:
        bias=None
        sol_dict={}
        B = z_old @ w_old.T
        A = z_new
        
        B_dict = norm_st(z_old,args)
        A_dict = norm_st(A,args)
        B = B_dict['tensor']@ w_old.T
        A = A_dict['tensor']
        R = B.T @ A
        U, S, V = torch.svd(R)
        
        # S[S<S.max()*1e-10]=0.
        # S[S>0]=1.
        # sol = U@torch.diag(S)@V.T
        
        # sol = U @ V.T
        if args.rotation:
            sv = torch.eye(S.shape[0], device=S.device)
            sv[-1,-1] = torch.det(U@V.T).sign()
            sol = U@sv@V.T
        else:
            sol = U@V.T
        if args.procrutes_norm_strategy == 'standard':
            A_diag = torch.diag(A_dict['std'].squeeze(0))
            B_diag = torch.diag(B_dict['std'].squeeze(0))
            # sol = sol * B_dict['std'].squeeze(0).unsqueeze(-1) / A_dict['std']
            sol_dict['W_in'] = B_diag @ sol @ A_diag.inverse()
            in_bias = B_dict['mean'] - A_dict['mean']@sol_dict['W_in'].transpose(0,1)
            sol_dict['in_bias'] = in_bias.squeeze(0)
            
            sol_dict['W_out'] = B_diag.inverse()@sol@A_diag
            sol_dict['out_bias'] = -sol_dict['in_bias']
        elif args.procrutes_norm_strategy == 'center':
            bias = B_dict['mean'] - A_dict['mean']@sol.transpose(0,1)
            sol_dict['W_in'] = sol
            sol_dict['in_bias'] = bias.squeeze(0)
            sol_dict['W_out'] = sol
            sol_dict['out_bias'] = -sol_dict['in_bias']
        else:
            sol_dict['W_in']=sol
            sol_dict['in_bias']=None
            sol_dict['W_out']=sol
            
            sol_dict['out_bias']=None
        return sol_dict
        # return sol, bias

def interpolate_orthogonal_matrices(Q0,Q1,t):
    # Q0 -> Q1 with t ratio
    R = Q0.T @ Q1
    numpy_R  = R.cpu().numpy()
    S = logm(numpy_R)
    St = S*t
    Rt = expm(St)
    Qt = Q0 @ torch.from_numpy(Rt).to(device=Q0.device,dtype=Q0.dtype)
    return Qt