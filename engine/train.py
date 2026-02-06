import os
import torch
import torch.nn as nn
import time
from datasets.common import get_dataloader, maybe_dictionarize
from models.loss import softmax_entropy

from engine.eval import evaluate_with_prelogit, eval_single_dataset
from models.noise import add_noise

import wandb

@torch.enable_grad()
def train_one_epoch(dataset, source_models, target_models,epoch, num_batches, scheduler,optimizer, loss_fn = nn.SmoothL1Loss(), print_every=20,args=None, logger=None,
                    do_eval=False):
    # model.train()
    
    source_zs_model,source_proxy_model = source_models
    target_zs_model,target_proxy_model = target_models
    
    source_zs_model.eval()
    source_proxy_model.eval()
    target_zs_model.eval()
    target_proxy_model.train()
    
    source_proxy_model.network[0].transfer_mode = True
    target_proxy_model.network[0].transfer_mode = True
    
    source_proxy_model.network[0].pre_compute_stats = args.source_stats
    target_proxy_model.network[0].pre_compute_stats = args.target_stats
    
    data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None,  is_transfer=True)
    
    for i, batch in enumerate(data_loader):
        start_time = time.time()
        
        step = i + epoch * num_batches
        if args.transfer_scheduler == 'cosine':
            scheduler(step)
        optimizer.zero_grad()

        batch = maybe_dictionarize(batch)
        if args.use_precompute_features:
            inputs = None
            labels = batch['labels'].to('cuda:0')
            source_feats = batch[f'{source_zs_model.image_encoder.model_name}_features'].to('cuda:0')
            target_feats = batch[f'{target_zs_model.image_encoder.model_name}_features'].to('cuda:0')
            source_feats /= source_feats.norm(dim=-1, keepdim=True)
            target_feats /= target_feats.norm(dim=-1, keepdim=True)
            source_logits = source_zs_model.classification_head(source_feats)
            target_logits = target_zs_model.classification_head(target_feats)
            data_time = time.time() - start_time
            with torch.no_grad():
                source_interm_feat = source_proxy_model(inputs,source_logits)['logits']
        else:
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time
            
            with torch.no_grad():
                source_logits = source_zs_model(inputs)
                target_logits = target_zs_model(inputs)
            
                source_interm_feat = source_proxy_model(inputs,source_logits)['logits']
        
        target_interm_feat = target_proxy_model(inputs,target_logits)['logits']
        
        loss = loss_fn(target_interm_feat,source_interm_feat)*args.transfer_loss_weight
        # loss = softmax_entropy(logit_dicts['logits']).mean(dim=0)
        # loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start_time

        if step % print_every == 0:
            percent_complete = 100 * i / len(data_loader)
            lr1 = [group['lr'] for group in optimizer.param_groups]
            lr = sum(lr1) / len(lr1)
            logger.info(
                f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loader)}]\tLR: {lr:.6f}\t"
                f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", 
            )
        if args.transfer_scheduler == 'step':
            scheduler.step()
    # logger.info(f"Epoch {epoch} done. Evaluating source transferred model {args.source_model}")
    # if args.eval_with_prelogits:
    #     source_proxy_model.network[0].transfer_mode = False
    #     source_proxy_model.network[0].transfer_out_proj = target_proxy_model.network[0].in_proj
    #     source_results = evaluate_with_prelogit(os.path.join(args.source_model_path,args.eval_datasets,'zero_shot')
    #                                         ,'pred_gt.json',args.eval_datasets,source_proxy_model, args, logger)
    # else:
    #     source_proxy_model.network[0].transfer_mode = False
    #     source_proxy_model.network[0].transfer_out_proj = target_proxy_model.network[0].in_proj
    #     source_results = eval_single_dataset([source_proxy_model,source_zs_model], args.eval_datasets, args)
    # source_proxy_model.network[0].transfer_out_proj = None
    # logger.info(str(args.source_model)+' for ' + ':' + str(source_results.get('top1')*100)+'%')
    
    logger.info(f"Epoch {epoch} done. Evaluating target model {args.target_model}")
    if args.eval_with_prelogits:
        target_proxy_model.network[0].transfer_mode = False
        target_results = evaluate_with_prelogit(os.path.join(args.target_model_path,args.eval_datasets,'zero_shot')
                                            ,'pred_gt.json',args.eval_datasets,target_proxy_model, args, logger)
    else:
        target_proxy_model.network[0].transfer_mode = False
        target_results = eval_single_dataset([target_proxy_model,target_zs_model], args.eval_datasets, args)
    logger.info(str(args.target_model)+' for ' + ':' + str(target_results.get('top1')*100)+'%')
    # return source_results, target_results
    return target_results

def post_train_one_epoch(dataset,target_models,epoch, num_batches, scheduler,optimizer, loss_fn = nn.CrossEntropyLoss(), print_every=20, transfer_noise_alpha=0.0,args=None, logger=None):
    target_zs_model, transfer_model, transfer_model_ema = target_models
    # if args.ldn_network == 'LogitAlone':
    #     transfer_model.network[0].pre_compute_stats = args.target_stats
    transfer_model.train()
    
    data_loader = get_dataloader(
        dataset, is_train=True, args=args, image_encoder=None, is_transfer=True)

    # total_training_time = 0
    epoch_start_time = time.time()
    for i, batch in enumerate(data_loader):
        start_time = time.time()
        
        step = i + epoch * num_batches
        if args.transfer_scheduler == 'cosine':
            scheduler(step)
        optimizer.zero_grad()

        batch = maybe_dictionarize(batch)
        if args.use_precompute_features:
            target_zs_inputs = batch[f'{target_zs_model.image_encoder.model_name}_features'].to(device=args.device)
            inputs = None
            labels = batch['labels'].to(device=args.device)
            data_time = time.time() - start_time
            with torch.no_grad():
                target_zs_model.eval()
                
                target_zs_inputs/=target_zs_inputs.norm(dim=-1,keepdim=True)
                target_zs_inputs = target_zs_model.classification_head(target_zs_inputs)
                
            if transfer_noise_alpha!=0:
                target_zs_inputs = add_noise(target_zs_inputs, args)
                
        else:
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            with torch.no_grad():
                target_zs_model.eval()
                target_zs_inputs = target_zs_model(inputs)
            
            # target_logits = target_model(inputs)
        
        
        logit_dicts = transfer_model(inputs, target_zs_inputs,ft_logit= source_ft_logits if args.use_interpolated_input else None)
        if 'interp_target' in logit_dicts:
            source_ft_logits = logit_dicts['interp_target']
        source_pred = logit_dicts['logits']
        if args.setting=='base2novel':
            # base class idx : dataset.base_class_idx                    
            loss = loss_fn(source_pred[:,dataset.base_class_idx], labels)
        else:
            num_compute_logit = source_pred.shape[1] if args.use_all_logits_for_loss else args.num_classes
            loss = loss_fn(source_pred[:,:num_compute_logit], labels)
        
        loss.backward()

        optimizer.step()
        batch_time = time.time() - start_time
        if transfer_model_ema is not None:
            transfer_model_ema.update(transfer_model)
        
        if step % print_every == 0:
            percent_complete = 100 * i / len(data_loader)
            lr1 = [group['lr'] for group in optimizer.param_groups]
            lr = sum(lr1) / len(lr1)
            train_log = f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loader)}]\tLR: {lr:.6f}\t"
            train_log += f"Loss: {loss.item():.6f}\t"
            train_log += f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"
            
            logger.info(

                train_log
            )
            if args.wandb:
                log_dict = {'loss':loss.item(),'lr':lr}
                if args.setting in ['few_shot','base2novel','cross_data','dg']:
                    log_dict =  {f'seed{args.seed}_post_trained_{k}': v for k, v in log_dict.items()}  
                wandb.log(log_dict)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_time = time.time() - epoch_start_time    
    # total_training_time += training_time   
    if args.scheduler == 'step':
        scheduler.step()
   
    return transfer_model, transfer_model_ema, training_time