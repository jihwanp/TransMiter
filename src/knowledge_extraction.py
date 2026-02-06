"""
TransMITER: Knowledge Extraction
Extract knowledge from fine-tuned source model via KL distillation.
"""

import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import copy
import json

from engine.args import get_base_parser, parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from datasets.templates import get_templates
from engine.eval import eval_single_dataset, eval_baseline
from models.modeling import ImageEncoder, ImageClassifier
from utils.utils import cosine_lr
from models.heads import get_classification_head
import wandb

from models.models import ProxyModel
from models.logit_stand import logit_stand
from models.stats import calculate_stats
from models.negative_class import negative_class_selection
from models.noise import add_noise

from datasets.few_shot import b2n_data_name_dict, data_name_dict

import geotorch


torch.backends.cudnn.benchmark = True


def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path + '/' + filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def build_models(args, dataset, logger):
    """Build source and target models."""
    # Build source model
    logger.info(f"Loading source model: {args.source_model}")
    args.model = args.source_model
    source_zs_image_encoder = ImageEncoder(args, keep_lang=False)
    source_ft_image_encoder = ImageEncoder(args, keep_lang=False)
    
    source_classification_head = get_classification_head(args, dataset, save_dir=args.source_model_path)
    source_zs_model = ImageClassifier(source_zs_image_encoder, source_classification_head)
    source_zs_model.freeze_head()
    
    # Build source fine-tuned models for each strategy
    source_ft_model = []
    if args.setting is not None:
        for ft_strategy in args.ft_strategy:
            theta_s = ImageClassifier(source_ft_image_encoder, copy.deepcopy(source_classification_head))
            if args.setting == 'base2novel':
                feat_dir_name = f'base2new_{ft_strategy}'
            elif args.setting == 'cross_data':
                feat_dir_name = f'crossdata_{ft_strategy}'
            elif args.setting == 'few_shot':
                feat_dir_name = f'fewshot_{ft_strategy}'
            elif args.setting == 'dg':
                feat_dir_name = f'dg_{ft_strategy}'
            else:
                raise NotImplementedError
            
            if 'features' in ft_strategy:
                text_pt_dir = os.path.join(args.data_location, feat_dir_name,
                                           data_name_dict[args.train_dataset],
                                           f'{args.seed}/16/test_features.pt')
            else:
                text_pt_dir = os.path.join(args.data_location, feat_dir_name,
                                           b2n_data_name_dict[args.train_dataset],
                                           f'{args.seed}/16/test_features.pt')
            try:
                text_feats = torch.load(text_pt_dir, map_location='cpu')[f'{args.source_model}_ft_text_features']
                text_feats /= text_feats.norm(dim=-1, keepdim=True)
                theta_s.classification_head.weight.data = text_feats * theta_s.image_encoder.model.logit_scale.exp()
                theta_s.classification_head.bias.data = torch.zeros_like(torch.randn(text_feats.shape[0])).to(dtype=text_feats.dtype)
            except:
                logger.info(f'Could not load text features for {args.setting} (model: {args.source_model})')
            theta_s.freeze_head()
            source_ft_model.append(theta_s)
    
    return source_zs_model, source_ft_model


def extract_knowledge(args, dataset, logger):
    """Main knowledge extraction function with KL distillation."""
    logger.info("=" * 60)
    logger.info("Phase 1: Knowledge Extraction (KL Distillation)")
    logger.info("=" * 60)
    
    # Negative class selection for padding
    args = negative_class_selection(dataset, args, logger)
    
    # Save auxiliary classes if enabled
    if args.save_auxiliary_classes and hasattr(args, 'selected_words') and args.selected_words:
        os.makedirs(args.seed_log_path, exist_ok=True)
        selected_words_path = os.path.join(args.seed_log_path, 'selected_words.json')
        with open(selected_words_path, 'w') as f:
            json.dump(args.selected_words, f, indent=2)
        logger.info(f"Saved auxiliary classes to {selected_words_path}")

    # Build models
    source_zs_model, source_ft_model = build_models(args, dataset, logger)
    
    # Move to device
    source_zs_model.to(args.device)
    for m in source_ft_model:
        m.to(args.device)
    
    # Get dataset info
    train_ds = get_dataset(
        dataset,
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        args=args
    )
    args.num_classes = len(train_ds.classnames)
    logger.info(f"Dataset: {dataset}, Number of classes: {args.num_classes}")

    # Build proxy model for knowledge extraction
    logger.info("Building proxy model...")
    model = ProxyModel(args).to(args.device)
    
    # Calculate statistics if needed
    if args.logit_norm_strategy in ['prestat', 'prestat_bn']:
        logger.info("Calculating logit statistics...")
        args.source_stats = calculate_stats(source_zs_model, dataset, args, args.source_model)
    else:
        args.source_stats = None
    
    model.network[0].pre_compute_stats = args.source_stats
    
    # Setup training
    data_loader = get_dataloader(
        train_ds, is_train=True, args=args, image_encoder=None, is_transfer=True
    )
    num_batches = len(data_loader)
    
    # Optimizer setup
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    logger.info(f"Number of trainable parameters: {num_params}")
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    # Scheduler setup
    if args.scheduler == 'cosine':
        args.warmup_length = args.epochs * num_batches * 0.05
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not recognized.")
    
    # Loss function setup (KL divergence)
    loss_function = getattr(args, 'loss_function', 'kl')
    if loss_function == 'kl':
        loss_fn = torch.nn.KLDivLoss(reduction='none') if args.logit_stand else torch.nn.KLDivLoss(reduction='batchmean')
    elif loss_function == 'kl_mean':
        loss_fn = torch.nn.KLDivLoss(reduction='mean')
    elif loss_function == 'smoothl1':
        loss_fn = torch.nn.SmoothL1Loss()
    elif loss_function == 'l2':
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    
    logger.info(f"Loss function: {loss_function}")
    logger.info(f"Starting KL distillation training for {args.epochs} epochs...")
    
    print_every = max(1, num_batches // 10)
    total_training_time = 0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        model.network[0].pre_compute_stats = args.source_stats
        
        data_loader = get_dataloader(
            train_ds, is_train=True, args=args, image_encoder=None, is_transfer=True
        )
        epoch_start_time = time.time()
        
        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            if args.scheduler == 'cosine':
                scheduler(step)
            optimizer.zero_grad()
            
            batch = maybe_dictionarize(batch)
            
            # Get precomputed features if available
            if args.use_precompute_features:
                if '__pretrained__' in args.source_model:
                    source_zs_inputs = batch[f'{args.source_model.split("__pretrained__")[0]}_features'].to(device=args.device)
                else:
                    source_zs_inputs = batch[f'{args.source_model}_features'].to(device=args.device)
                inputs = None
                labels = batch['labels'].to(device=args.device)
                data_time = time.time() - start_time
                
                with torch.no_grad():
                    source_zs_model.eval()
                    for mod in source_ft_model:
                        mod.eval()
                    
                    source_zs_inputs = source_zs_inputs / source_zs_inputs.norm(dim=-1, keepdim=True)
                    source_zs_logits = source_zs_model.classification_head(source_zs_inputs)
                    
                    # Get finetuned logits
                    all_source_ft_logits = []
                    if args.use_ft_logits:
                        if '__pretrained__' in args.source_model:
                            source_ft_inputs = batch[f'{args.source_model.split("__pretrained__")[0]}_ft_logits']
                        else:
                            source_ft_inputs = batch[f'{args.source_model}_ft_logits']
                        for ft_mod, ft_inp in zip(source_ft_model, source_ft_inputs):
                            ft_logit = ft_inp.to(device=args.device)
                            all_source_ft_logits.append(ft_logit)
                    else:
                        source_ft_inputs = batch[f'{args.source_model}_ft_features']
                        for ft_mod, ft_inp in zip(source_ft_model, source_ft_inputs):
                            ft_inp = ft_inp.to(device=args.device)
                            ft_inp = ft_inp / ft_inp.norm(dim=-1, keepdim=True)
                            source_ft_logits = ft_mod.classification_head(ft_inp)
                            all_source_ft_logits.append(source_ft_logits)
                    
                    source_ft_logits = torch.stack(all_source_ft_logits)
                    if getattr(args, 'apply_tf_ensemble', False):
                        prob = source_ft_logits.softmax(dim=-1)
                        max_prob_per_model = prob.max(dim=-1, keepdim=True)[0]
                        weight_per_model = max_prob_per_model.softmax(dim=0)
                        source_ft_logits = (weight_per_model * source_ft_logits).sum(dim=0)
                    else:
                        source_ft_logits = source_ft_logits.mean(dim=0)
                
                if args.noise_alpha != 0:
                    source_zs_logits = add_noise(source_zs_logits, args)
            else:
                inputs = batch['images'].to(args.device)
                labels = batch['labels'].to(args.device)
                data_time = time.time() - start_time
                
                with torch.no_grad():
                    source_zs_model.eval()
                    for mod in source_ft_model:
                        mod.eval()
                    source_zs_logits = source_zs_model(inputs)
                    
                    all_source_ft_logits = []
                    for ft_mod in source_ft_model:
                        source_ft_logit = ft_mod(inputs)
                        all_source_ft_logits.append(source_ft_logit)
                    source_ft_logits = torch.stack(all_source_ft_logits).mean(dim=0)
            
            # Forward pass through proxy model
            logit_dicts = model(inputs, source_zs_logits, ft_logit=source_ft_logits if getattr(args, 'use_interpolated_input', False) else None)
            if 'interp_target' in logit_dicts:
                source_ft_logits = logit_dicts['interp_target']
            source_pred = logit_dicts['logits']
            
            # Compute KL distillation loss
            tau = args.temperature
            source_ft_logits = source_ft_logits.to(dtype=source_pred.dtype)
            
            if args.logit_stand:
                num_compute_logit = source_pred.shape[1] if args.use_all_logits_for_loss else args.num_classes
                source_ft_logits_stand, _, ft_std = logit_stand(source_ft_logits[:, :num_compute_logit], return_others=True)
                source_pred_stand, _, pt_std = logit_stand(source_pred[:, :num_compute_logit], return_others=True)
                
                loss_scale = tau * tau * getattr(args, 'teacher_temperature', 1.0) * getattr(args, 'main_loss_coef', 1.0) * ft_std * pt_std
                loss = loss_fn(
                    F.log_softmax(source_pred_stand / (tau * pt_std), dim=-1),
                    F.softmax(source_ft_logits_stand / (tau * getattr(args, 'teacher_temperature', 1.0) * ft_std), dim=-1)
                )
                loss *= loss_scale
                loss = loss.sum(dim=-1).mean()
            else:
                num_compute_logit = source_pred.shape[1] if args.use_all_logits_for_loss else args.num_classes
                loss_scale = tau * tau * getattr(args, 'teacher_temperature', 1.0) * getattr(args, 'main_loss_coef', 1.0)
                loss = loss_fn(
                    F.log_softmax(source_pred[:, :num_compute_logit] / (tau), dim=-1),
                    F.softmax(source_ft_logits[:, :num_compute_logit] / (tau * getattr(args, 'teacher_temperature', 1.0)), dim=-1)
                ) * loss_scale
            
            # Additional MSE loss if enabled
            if getattr(args, 'use_additional_mse_loss', False):
                mse_loss = getattr(args, 'l2_loss_coef', 1.0) * F.mse_loss(
                    source_pred[:, :num_compute_logit] / 100,
                    source_ft_logits[:, :num_compute_logit] / 100
                )
                loss += mse_loss
            
            loss.backward()
            optimizer.step()
            batch_time = time.time() - start_time
            
            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                lr1 = [group['lr'] for group in optimizer.param_groups]
                lr = sum(lr1) / len(lr1)
                train_log = f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loader)}]\tLR: {lr:.6f}\t"
                train_log += f"Loss: {loss.item():.6f}\t"
                train_log += f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                logger.info(train_log)
                
                if args.wandb:
                    log_dict = {'loss': loss.item(), 'lr': lr}
                    if args.setting in ['few_shot', 'base2novel', 'cross_data', 'dg']:
                        log_dict = {f'seed{args.seed}_{k}': v for k, v in log_dict.items()}
                    wandb.log(log_dict)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_time = time.time() - epoch_start_time
        total_training_time += training_time
        
        if args.scheduler == 'step':
            scheduler.step()
        
        # Evaluation
        logger.info(f"Epoch {epoch} done. Evaluating...")
        model.network[0].pre_compute_stats = args.source_stats
        source_results = eval_single_dataset([model, source_zs_model], args.eval_datasets, args)
        
        if args.setting == 'base2novel':
            base_acc = source_results.get('top1_base', 0) * 100
            novel_acc = source_results.get('top1_novel', 0) * 100
            logger.info(f"Source {args.source_model} {args.eval_datasets}: Base {base_acc:.2f}%, Novel {novel_acc:.2f}%")
        else:
            logger.info(f"{args.source_model} for {args.eval_datasets}: {source_results.get('top1', 0) * 100:.2f}%")
    
    logger.info(f"Total training time: {total_training_time:.2f}s")
    
    # Save extracted model - seed_log_path already includes ft_strategy
    save_dir = args.seed_log_path
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'proxy_model.pt'))
    
    # Save config
    config = {
        'source_model': args.source_model,
        'dataset': dataset,
        'proj_dim': args.proj_dim,
        'num_classes': args.num_classes,
        'ft_strategy': args.ft_strategy,
        'seed': args.seed,
        'epochs': args.epochs,
        'lr': args.lr,
        'temperature': args.temperature,
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved extracted knowledge to {save_dir}")
    
    # Final evaluation
    eval_result = eval_single_dataset([model, source_zs_model], dataset, args)
    base_acc = eval_result.get('top1_base', eval_result.get('top1', 0)) * 100
    novel_acc = eval_result.get('top1_novel', 0) * 100
    
    if args.setting == 'base2novel':
        hm = 2 * base_acc * novel_acc / (base_acc + novel_acc) if (base_acc + novel_acc) > 0 else 0
        logger.info(f"Final Results - Base: {base_acc:.2f}%, Novel: {novel_acc:.2f}%, HM: {hm:.2f}%")
    else:
        logger.info(f"Final Results - Accuracy: {base_acc:.2f}%")
    
    return {
        'save_dir': save_dir,
        'base_acc': base_acc,
        'novel_acc': novel_acc,
        'model': model,
    }


def add_extraction_arguments(parser):
    """Add knowledge extraction specific arguments."""
    parser.add_argument('--save_auxiliary_classes', action='store_true', help='Save auxiliary classes')
    parser.add_argument('--loss_function', type=str, default='kl', choices=['kl', 'kl_mean', 'smoothl1', 'l2'], help='Loss function for distillation')
    parser.add_argument('--teacher_temperature', type=float, default=1.0, help='Teacher temperature for KL distillation')
    parser.add_argument('--main_loss_coef', type=float, default=1.0, help='Main loss coefficient')
    parser.add_argument('--use_additional_mse_loss', action='store_true', help='Use additional MSE loss')
    parser.add_argument('--l2_loss_coef', type=float, default=1.0, help='L2/MSE loss coefficient')
    return parser


def main():
    parser = get_base_parser()
    parser = add_extraction_arguments(parser)
    args = parse_arguments(parser)
    set_seed(args)
    
    # Use --dataset as the target dataset
    dataset = args.dataset
    args.train_dataset = dataset
    args.eval_datasets = dataset
    
    # Setup logging - include ft_strategy in path
    exp_name = f"KET_{args.source_model}"
    ft_strategy_str = '_'.join(args.ft_strategy)
    log_dir = f"./logs/{exp_name}/{dataset}"
    args.seed_log_path = os.path.join(log_dir, str(args.seed), ft_strategy_str)
    logger = create_log_dir(args.seed_log_path)
    
    logger.info("=" * 60)
    logger.info("TransMITER: Knowledge Extraction (KL Distillation)")
    logger.info("=" * 60)
    logger.info(f"Source Model: {args.source_model}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Setting: {args.setting}")
    logger.info(f"FT Strategy: {args.ft_strategy}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group if hasattr(args, 'wandb_group') else 'knowledge_extraction',
            name=f"KET_{args.source_model}_{dataset}_seed{args.seed}",
            config=vars(args)
        )
    
    result = extract_knowledge(args, dataset, logger)
    
    if args.setting == 'base2novel':
        logger.info(f"\nFinal: {dataset}: Base={result['base_acc']:.2f}%, Novel={result['novel_acc']:.2f}%")
    else:
        logger.info(f"\nFinal: {dataset}: Acc={result['base_acc']:.2f}%")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
