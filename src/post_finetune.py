"""
TransMITER: Post-Training with Labels
Fine-tune the transferred model using labeled data for improved performance.
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
from engine.train import train_one_epoch
from models.modeling import ImageEncoder, ImageClassifier
from utils.utils import cosine_lr
from models.heads import get_classification_head
import wandb

from models.models import ProxyModel, TransferProxy
from models.logit_stand import logit_stand
from models.stats import calculate_stats
from models.negative_class import negative_class_selection

from datasets.few_shot import b2n_data_name_dict, data_name_dict, FewshotDataset
from datasets.feature_data import FeatureDataset

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
    
    # Build source fine-tuned models
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
    
    # Build target model
    logger.info(f"Loading target model: {args.target_model}")
    args.model = args.target_model
    target_image_encoder = ImageEncoder(args, keep_lang=False)
    
    target_classification_head = get_classification_head(args, dataset, save_dir=args.target_model_path)
    target_model = ImageClassifier(target_image_encoder, target_classification_head)
    target_model.freeze_head()
    
    return source_zs_model, source_ft_model, target_model


def load_transferred_model(args, dataset, logger):
    """Load pre-trained transferred model."""
    # Construct path to transferred model
    reg_coef = args.reg_procrutes_coef[0] if isinstance(args.reg_procrutes_coef, list) else args.reg_procrutes_coef
    
    model_dir = os.path.join(
        args.source_model_path.replace(args.source_model, f'{args.source_model}->{args.target_model}'),
        dataset,
        f'reg_{reg_coef}'
    )
    for ft_st in args.ft_strategy:
        model_dir = os.path.join(model_dir, ft_st)
    
    model_path = os.path.join(model_dir, 'proxy_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transferred model not found at {model_path}. Run knowledge_transfer.py first.")
    
    logger.info(f"Loading transferred model from {model_path}")
    
    # Build proxy model and load weights
    proxy_model = ProxyModel(args)
    proxy_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return proxy_model, model_dir


def get_train_dataloader(args, dataset, logger):
    """Get training dataloader for post-training."""
    if args.use_precompute_features:
        # Use precomputed features
        if args.setting == 'base2novel':
            feat_dir_name = f'base2new_{args.ft_strategy[0]}'
        elif args.setting == 'cross_data':
            feat_dir_name = f'crossdata_{args.ft_strategy[0]}'
        elif args.setting == 'few_shot':
            feat_dir_name = f'fewshot_{args.ft_strategy[0]}'
        elif args.setting == 'dg':
            feat_dir_name = f'dg_{args.ft_strategy[0]}'
        else:
            feat_dir_name = args.ft_strategy[0]
        
        if 'features' in args.ft_strategy[0]:
            feat_dir = os.path.join(args.data_location, feat_dir_name,
                                    data_name_dict.get(dataset, dataset),
                                    f'{args.seed}/16/train_features.pt')
        else:
            feat_dir = os.path.join(args.data_location, feat_dir_name,
                                    b2n_data_name_dict.get(dataset, dataset),
                                    f'{args.seed}/16/train_features.pt')
        
        logger.info(f"Loading precomputed features from {feat_dir}")
        train_ds = FeatureDataset(feat_dir, args.target_model)
    else:
        # Use raw images with FewshotDataset
        train_ds = FewshotDataset(
            dataset,
            args.data_location,
            args.seed,
            num_shots=16,
            split='train',
            args=args
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.transfer_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True if len(train_ds) > args.transfer_batch_size else False
    )
    
    return train_loader


def post_finetune(args, dataset, logger):
    """Post-training with labels."""
    logger.info("=" * 60)
    logger.info("Phase 3: Post-Training with Labels")
    logger.info("=" * 60)
    
    # Load saved auxiliary classes or run negative class selection
    saved_words_path = os.path.join(args.seed_log_path, 'selected_words.json')
    if os.path.exists(saved_words_path):
        with open(saved_words_path, 'r') as f:
            args.selected_words = json.load(f)
        logger.info(f"Loaded auxiliary classes from {saved_words_path}")
        # Still need to run for other setup (e.g., num_pad)
        args = negative_class_selection(dataset, args, logger, use_saved=True)
    else:
        args = negative_class_selection(dataset, args, logger)

    # Build models
    source_zs_model, source_ft_model, target_model = build_models(args, dataset, logger)
    
    # Move to device
    source_zs_model.to(args.device)
    for m in source_ft_model:
        m.to(args.device)
    target_model.to(args.device)
    
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
    
    # Load transferred model
    proxy_model, model_dir = load_transferred_model(args, dataset, logger)
    proxy_model.to(args.device)
    
    # Calculate statistics
    if args.logit_norm_strategy in ['prestat', 'prestat_bn']:
        args.target_stats = calculate_stats(target_model, dataset, args, args.target_model)
    else:
        args.target_stats = None
    proxy_model.network[0].pre_compute_stats = args.target_stats
    
    # Get training dataloader
    train_loader = get_train_dataloader(args, dataset, logger)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    
    # Remove orthogonality constraints for fine-tuning
    if args.remove_orthogonality:
        logger.info("Removing orthogonality constraints for fine-tuning")
        for module in proxy_model.modules():
            if hasattr(module, 'parametrizations'):
                geotorch.reals(module, 'weight')
    
    # Setup optimizer
    if args.transfer_optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            proxy_model.parameters(),
            lr=args.transfer_lr,
            weight_decay=args.transfer_wd if hasattr(args, 'transfer_wd') else 0.01
        )
    elif args.transfer_optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            proxy_model.parameters(),
            lr=args.transfer_lr,
            momentum=0.9,
            weight_decay=args.transfer_wd if hasattr(args, 'transfer_wd') else 1e-4
        )
    else:
        optimizer = torch.optim.Adam(
            proxy_model.parameters(),
            lr=args.transfer_lr
        )
    
    # Setup scheduler
    num_steps = args.transfer_epochs * len(train_loader)
    scheduler = cosine_lr(optimizer, args.transfer_lr, num_steps // 10, num_steps)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_hm = 0
    best_state = None
    best_epoch = 0
    
    for epoch in range(args.transfer_epochs):
        proxy_model.train()
        target_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            scheduler(step)
            
            if args.use_precompute_features:
                features = batch['features'].to(args.device)
                labels = batch['labels'].to(args.device)
            else:
                batch = maybe_dictionarize(batch)
                images = batch['images'].to(args.device)
                labels = batch['labels'].to(args.device)
                
                # Extract features with target model
                with torch.no_grad():
                    features = target_model.image_encoder(images)
            
            # Add noise if specified
            if hasattr(args, 'transfer_noise_alpha') and args.transfer_noise_alpha > 0:
                noise = torch.randn_like(features) * args.transfer_noise_alpha
                features = features + noise
            
            # Forward through proxy model
            with torch.no_grad():
                target_logits = target_model.classification_head(features)
            
            # Apply logit normalization if needed
            if args.logit_norm_strategy != 'none':
                target_logits = logit_stand(target_logits, args.target_stats, args.logit_norm_strategy)
            
            # Get transferred logits
            transferred_logits = proxy_model(target_logits)
            
            # Compute loss
            loss = criterion(transferred_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{args.transfer_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate
        proxy_model.eval()
        eval_result = eval_single_dataset(
            [proxy_model, target_model],
            dataset, args
        )
        
        base_acc = eval_result.get('top1_base', eval_result.get('top1', 0)) * 100
        novel_acc = eval_result.get('top1_novel', 0) * 100
        
        if args.setting == 'base2novel':
            hm = 2 * base_acc * novel_acc / (base_acc + novel_acc) if (base_acc + novel_acc) > 0 else 0
            logger.info(f"Epoch {epoch+1}: Base={base_acc:.2f}%, Novel={novel_acc:.2f}%, HM={hm:.2f}%")
            
            if hm > best_hm:
                best_hm = hm
                best_state = copy.deepcopy(proxy_model.state_dict())
                best_epoch = epoch + 1
                best_base = base_acc
                best_novel = novel_acc
        else:
            logger.info(f"Epoch {epoch+1}: Accuracy={base_acc:.2f}%")
            if base_acc > best_hm:
                best_hm = base_acc
                best_state = copy.deepcopy(proxy_model.state_dict())
                best_epoch = epoch + 1
    
    # Save best model
    if best_state is not None:
        proxy_model.load_state_dict(best_state)
        
        save_dir = os.path.join(model_dir, 'post_trained')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, os.path.join(save_dir, 'proxy_model.pt'))
        
        # Save config
        config = {
            'source_model': args.source_model,
            'target_model': args.target_model,
            'dataset': dataset,
            'transfer_epochs': args.transfer_epochs,
            'transfer_lr': args.transfer_lr,
            'transfer_batch_size': args.transfer_batch_size,
            'transfer_optimizer': args.transfer_optimizer,
            'best_epoch': best_epoch,
            'seed': args.seed,
        }
        
        if args.setting == 'base2novel':
            config['best_base_acc'] = best_base
            config['best_novel_acc'] = best_novel
            config['best_hm'] = best_hm
        else:
            config['best_accuracy'] = best_hm
        
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"\nBest model saved to {save_dir}")
        logger.info(f"Best epoch: {best_epoch}")
    
    # Alpha tuning after post-training
    if args.alpha_tuning_after_post_train:
        logger.info("\nPerforming alpha tuning...")
        best_alpha, best_result = alpha_tuning(proxy_model, target_model, dataset, args, logger)
        
        return {
            'best_hm': best_hm,
            'best_epoch': best_epoch,
            'best_alpha': best_alpha,
            'alpha_result': best_result,
        }
    
    return {
        'best_hm': best_hm,
        'best_epoch': best_epoch,
    }


def alpha_tuning(proxy_model, target_model, dataset, args, logger):
    """Tune ensemble alpha between proxy and target model."""
    proxy_model.eval()
    target_model.eval()
    
    best_alpha = 0.5
    best_hm = 0
    best_result = None
    
    for alpha in np.arange(0.0, 1.05, 0.1):
        args.ensemble_alpha = alpha
        
        eval_result = eval_single_dataset(
            [proxy_model, target_model],
            dataset, args,
            ensemble_alpha=alpha
        )
        
        base_acc = eval_result.get('top1_base', eval_result.get('top1', 0)) * 100
        novel_acc = eval_result.get('top1_novel', 0) * 100
        
        if args.setting == 'base2novel':
            hm = 2 * base_acc * novel_acc / (base_acc + novel_acc) if (base_acc + novel_acc) > 0 else 0
            logger.info(f"Alpha={alpha:.1f}: Base={base_acc:.2f}%, Novel={novel_acc:.2f}%, HM={hm:.2f}%")
            
            if hm > best_hm:
                best_hm = hm
                best_alpha = alpha
                best_result = {'base': base_acc, 'novel': novel_acc, 'hm': hm}
        else:
            logger.info(f"Alpha={alpha:.1f}: Accuracy={base_acc:.2f}%")
            if base_acc > best_hm:
                best_hm = base_acc
                best_alpha = alpha
                best_result = {'accuracy': base_acc}
    
    logger.info(f"\nBest alpha: {best_alpha}")
    return best_alpha, best_result


def add_post_finetune_arguments(parser):
    """Add post-finetune specific arguments."""
    # Post-training specific arguments
    parser.add_argument('--transfer_epochs', type=int, default=0, help='Number of epochs for post-training')
    parser.add_argument('--transfer_lr', type=float, default=1e-4, help='Learning rate for post-training')
    parser.add_argument('--transfer_wd', type=float, default=0.001, help='Weight decay for post-training')
    parser.add_argument('--transfer_noise_alpha', type=float, default=0.0, help='Noise alpha for post-training')
    parser.add_argument('--transfer_batch_size', type=int, default=128, help='Batch size for post-training')
    parser.add_argument('--transfer_scheduler', type=str, default='cosine', choices=['cosine', 'step','none'], help='LR scheduler')
    parser.add_argument('--transfer_step_size', type=int, default=30, help='Step size for step scheduler')
    parser.add_argument('--transfer_loss_weight', type=float, default=5.0, help='Loss weight for post-training')
    parser.add_argument('--transfer_optimizer', type=str, default='adamw', choices=['adamw','sgd'], help='Optimizer')
    parser.add_argument('--remove_orthogonality', action='store_true', help='Remove orthogonality for post-training')
    parser.add_argument('--no_bias', action='store_true', help='No bias for post-training')
    parser.add_argument('--use_post_train_ema', action='store_true', help='Use EMA for post-training')
    parser.add_argument('--post_train_ema_decay', type=float, default=0.9, help='EMA decay')
    parser.add_argument('--no_lr_decay_for_tm', action='store_true', help='No LR decay for transition matrix')
    parser.add_argument('--alpha_tuning_after_post_train', action='store_true', help='Alpha tuning after post-train')
    parser.add_argument('--save_auxiliary_classes', action='store_true', help='Save auxiliary classes')
    return parser


def main():
    parser = get_base_parser()
    parser = add_post_finetune_arguments(parser)
    args = parse_arguments(parser)
    set_seed(args)
    
    # Use --dataset as the target dataset
    dataset = args.dataset
    args.train_dataset = dataset
    args.eval_datasets = dataset
    
    # Setup logging
    exp_name = f"PostTrain_{args.source_model}->{args.target_model}"
    log_dir = f"./logs/{exp_name}/{dataset}"
    args.seed_log_path = os.path.join(log_dir, str(args.seed))
    logger = create_log_dir(args.seed_log_path)
    
    logger.info("=" * 60)
    logger.info("TransMITER: Post-Training with Labels")
    logger.info("=" * 60)
    logger.info(f"Source Model: {args.source_model}")
    logger.info(f"Target Model: {args.target_model}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Setting: {args.setting}")
    logger.info(f"FT Strategy: {args.ft_strategy}")
    logger.info(f"Transfer Epochs: {args.transfer_epochs}")
    logger.info(f"Transfer LR: {args.transfer_lr}")
    logger.info(f"Transfer Batch Size: {args.transfer_batch_size}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group if hasattr(args, 'wandb_group') else 'post_training',
            name=f"PostTrain_{args.source_model}->{args.target_model}_{dataset}_seed{args.seed}",
            config=vars(args)
        )
    
    result = post_finetune(args, dataset, logger)
    
    if args.setting == 'base2novel':
        logger.info(f"\nFinal: {dataset}: Best HM={result['best_hm']:.2f}% (Epoch {result['best_epoch']})")
    else:
        logger.info(f"\nFinal: {dataset}: Best Acc={result.get('best_acc', result.get('best_hm', 0)):.2f}%")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
