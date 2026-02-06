"""
TransMITER: Knowledge Transfer
Load extracted knowledge and apply to target model with different regularization coefficients.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn
import random
import numpy as np
import json
import copy
import gc

from engine.args import parse_arguments
from datasets.registry import get_dataset
from engine.eval import eval_single_dataset, eval_baseline
from models.modeling import ImageEncoder, ImageClassifier
from models.heads import get_classification_head
import wandb

from models.models import ProxyModel
from models.stats import calculate_stats
from engine.ortho_proc import run_models_ver2, orthogonal_procrutes
from models.negative_class import negative_class_selection


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


def transfer_knowledge(args, dataset, logger):
    """Apply knowledge transfer with different regularization coefficients."""
    logger.info("=" * 60)
    logger.info("Phase 2: Knowledge Transfer")
    logger.info("=" * 60)
    
    # Load saved auxiliary classes or run negative class selection
    ft_strategy_str = '_'.join(args.ft_strategy)
    saved_words_path = os.path.join(f"logs/KET_{args.source_model}", args.dataset, str(args.seed), ft_strategy_str, 'selected_words.json')
    if os.path.exists(saved_words_path):
        with open(saved_words_path, 'r') as f:
            args.selected_words = json.load(f)
        logger.info(f"Loaded auxiliary classes from {saved_words_path}")
        # Still need to run for other setup (e.g., num_pad)
        args = negative_class_selection(dataset, args, logger, use_saved=True)
    else:
        args = negative_class_selection(dataset, args, logger)
    
    # Build source zero-shot model (needed for run_models_ver2)
    logger.info(f"Loading source model: {args.source_model}")
    args.model = args.source_model
    source_zs_image_encoder = ImageEncoder(args, keep_lang=False)
    source_classification_head = get_classification_head(args, dataset, save_dir=args.source_model_path)
    source_zs_model = ImageClassifier(source_zs_image_encoder, source_classification_head)
    source_zs_model.freeze_head()
    source_zs_model.to(args.device)
    
    
    # Build target model
    logger.info(f"Loading target model: {args.target_model}")
    args.model = args.target_model
    target_image_encoder = ImageEncoder(args, keep_lang=False)
    target_classification_head = get_classification_head(args, dataset, save_dir=args.target_model_path)
    target_model = ImageClassifier(target_image_encoder, target_classification_head)
    target_model.freeze_head()
    target_model.to(args.device)
    
    eval_baseline(target_model,dataset,args,is_base_finetuned=False)
    
    # Get dataset info
    preprocess_fn = transforms.Compose(source_zs_model.val_preprocess.transforms[:-1]) if args.post_normalize else source_zs_model.val_preprocess
    train_ds = get_dataset(
            args.train_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=256,
            is_transfer=True,
            use_train_for_proxy=args.use_train_for_proxy,
            data_ratio=args.data_ratio,
            args=args
        )
    args.num_classes = len(train_ds.classnames)
    logger.info(f"Dataset: {dataset}, Number of classes: {args.num_classes}")
    
    # Load extracted proxy model from knowledge_extraction
    ket_dir = f"KET_{args.source_model}"
    extraction_dir = os.path.join("./logs", ket_dir, args.dataset, str(args.seed), ft_strategy_str)
    extracted_model_path = os.path.join(extraction_dir, 'proxy_model.pt')
    if not os.path.exists(extracted_model_path):
        raise FileNotFoundError(f"Extracted model not found at {extracted_model_path}. Run knowledge_extraction.py first.")

    logger.info(f"Loading extracted proxy model from {extracted_model_path}")
    source_proxy_model = ProxyModel(args).to(args.device)
    ckpt = torch.load(extracted_model_path)
    source_proxy_model.load_state_dict(ckpt)
    
    parametrize.remove_parametrizations(source_proxy_model.network[0].in_proj,'weight')
    # source_proxy_model.network[0].transfer_out_proj = nn.Linear(source_proxy_model.network[0].in_proj.weight.shape[1],source_proxy_model.network[0].in_proj.weight.shape[0],bias=False).to(device=args.device)
    # source_proxy_model.network[0].transfer_out_proj.weight.data = source_proxy_model.network[0].in_proj.weight.data
    source_proxy_model.eval()
    
    # Calculate statistics
    if args.logit_norm_strategy in ['prestat', 'prestat_bn']:
        args.source_stats = calculate_stats(source_zs_model, dataset, args, args.source_model)
        args.target_stats = calculate_stats(target_model, dataset, args, args.target_model)
    else:
        args.source_stats = None
        args.target_stats = None
    
    source_proxy_model.network[0].pre_compute_stats = args.source_stats
    
    # Evaluate loaded source proxy model to verify it was loaded correctly
    logger.info("Evaluating loaded source proxy model...")
    source_eval_result = eval_single_dataset(
        [source_proxy_model, source_zs_model],
        dataset, args
    )
    if args.setting == 'base2novel':
        src_base = source_eval_result.get('top1_base', 0) * 100
        src_novel = source_eval_result.get('top1_novel', 0) * 100
        src_hm = 2 * src_base * src_novel / (src_base + src_novel) if (src_base + src_novel) > 0 else 0
        logger.info(f"Source proxy model: Base={src_base:.2f}%, Novel={src_novel:.2f}%, HM={src_hm:.2f}%")
    else:
        logger.info(f"Source proxy model: Acc={source_eval_result.get('top1', 0) * 100:.2f}%")
    
    # Copy source proxy model to create target proxy model (via state_dict for parametrized modules)
    logger.info("Copying source proxy model to target proxy model...")
    # target_proxy_model = ProxyModel(args).to(args.device)
    # target_proxy_model.load_state_dict(ckpt)
    target_proxy_model = copy.deepcopy(source_proxy_model)
    target_proxy_model.network[0].pre_compute_stats = args.target_stats
    
    
    # Evaluate target proxy model before basis change
    logger.info("Evaluating target proxy model (before basis change)...")
    target_eval_result = eval_single_dataset(
        [target_proxy_model, target_model],
        dataset, args
    )
    if args.setting == 'base2novel':
        tgt_base = target_eval_result.get('top1_base', 0) * 100
        tgt_novel = target_eval_result.get('top1_novel', 0) * 100
        tgt_hm = 2 * tgt_base * tgt_novel / (tgt_base + tgt_novel) if (tgt_base + tgt_novel) > 0 else 0
        logger.info(f"[Before basis change] Target proxy model: Base={tgt_base:.2f}%, Novel={tgt_novel:.2f}%, HM={tgt_hm:.2f}%")
    else:
        logger.info(f"[Before basis change] Target proxy model: Acc={target_eval_result.get('top1', 0) * 100:.2f}%")
    
    # Run models to get latent representations for orthogonal Procrustes
    logger.info("Extracting latent representations for Procrustes alignment...")
    w_old = source_proxy_model.network[0].in_proj.weight.data
    target_feats, source_feats = run_models_ver2(
        [source_zs_model, source_proxy_model],
        [target_model, target_proxy_model],
        train_ds, 
        args, 
        w_old=w_old if args.latent_procrutes and args.extract_latent_place=='none' else None
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    reg_coef = args.reg_procrutes_coef
    logger.info(f"\nApplying reg_procrutes_coef = {reg_coef}")
    
    # Orthogonal Procrustes alignment
    proc_result = orthogonal_procrutes(target_feats, source_feats, w_old, args)
    
    # Create test proxy model
    test_proxy_model = copy.deepcopy(target_proxy_model)
    test_proxy_model.network[0].transfer_procrutes_mode = False
    test_proxy_model.network[0].pre_compute_stats = args.target_stats
    
    # Apply alignment            
    # if proc_result['W_in'] is not None:
    #     test_proxy_model.network[0].in_proj.weight.data = proc_result['W_in'].to(args.device)
    # if proc_result['W_out'] is not None:
    #     orth_out_proj = copy.deepcopy(test_proxy_model.network[0].in_proj)
    #     orth_out_proj.weight.data = proc_result['W_out'].to(args.device)
    #     test_proxy_model.network[0].transfer_out_proj = orth_out_proj
    
    orth_in_proj = nn.Linear(test_proxy_model.network[0].in_proj.weight.shape[1],test_proxy_model.network[0].in_proj.weight.shape[0],bias=True,device=args.device)
    orth_in_proj.bias.data.zero_()
    orth_out_proj = copy.deepcopy(orth_in_proj)
    if proc_result['W_in'] is not None:
        orth_in_proj.weight.data = proc_result['W_in'].to(args.device)
    if proc_result['W_out'] is not None:
        orth_out_proj.weight.data = proc_result['W_out'].to(args.device)
    test_proxy_model.network[0].in_proj = orth_in_proj
    test_proxy_model.network[0].transfer_out_proj = orth_out_proj
    
    # Evaluate
    eval_result = eval_single_dataset(
        [test_proxy_model, target_model],
        dataset, args
    )
    
    base_acc = eval_result.get('top1_base', eval_result.get('top1', 0)) * 100
    novel_acc = eval_result.get('top1_novel', 0) * 100
    
    if args.setting == 'base2novel':
        hm = 2 * base_acc * novel_acc / (base_acc + novel_acc) if (base_acc + novel_acc) > 0 else 0
        logger.info(f"reg={reg_coef}: Base={base_acc:.2f}%, Novel={novel_acc:.2f}%, HM={hm:.2f}%")
        result = {
            'reg_coef': reg_coef,
            'base_acc': base_acc,
            'novel_acc': novel_acc,
            'hm': hm,
        }
    else:
        logger.info(f"reg={reg_coef}: Accuracy={base_acc:.2f}%")
        result = {
            'reg_coef': reg_coef,
            'accuracy': base_acc,
        }
    
    # Save model - seed_log_path already includes ft_strategy
    save_dir = args.seed_log_path
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(test_proxy_model.state_dict(), os.path.join(save_dir, 'proxy_model.pt'))
    
    # Save config
    config = {
        'source_model': args.source_model,
        'target_model': args.target_model,
        'dataset': dataset,
        'reg_coef': reg_coef,
        'proj_dim': args.proj_dim,
        'num_classes': args.num_classes,
        'ft_strategy': args.ft_strategy,
        'seed': args.seed,
        'base_acc': result.get('base_acc', result.get('accuracy')),
        'novel_acc': result.get('novel_acc', 0),
        'hm': result.get('hm', result.get('accuracy')),
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\nModel saved to {save_dir}")
    
    return {
        'result': result,
    }


def main():
    args = parse_arguments()
    set_seed(args)
    
    # Use --dataset as the target dataset
    dataset = args.dataset
    args.train_dataset = dataset
    args.eval_datasets = dataset
    
    # Setup logging - include ft_strategy in path
    exp_name = f"Transfer_{args.source_model}->{args.target_model}"
    ft_strategy_str = '_'.join(args.ft_strategy)
    log_dir = f"./logs/{exp_name}/{dataset}"
    args.seed_log_path = os.path.join(log_dir, str(args.seed), ft_strategy_str)
    logger = create_log_dir(args.seed_log_path)
    
    logger.info("=" * 60)
    logger.info("TransMITER: Knowledge Transfer")
    logger.info("=" * 60)
    logger.info(f"Source Model: {args.source_model}")
    logger.info(f"Target Model: {args.target_model}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Setting: {args.setting}")
    logger.info(f"FT Strategy: {args.ft_strategy}")
    logger.info(f"Reg Coefficient: {args.reg_procrutes_coef}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group if hasattr(args, 'wandb_group') else 'knowledge_transfer',
            name=f"Transfer_{args.source_model}->{args.target_model}_{dataset}_seed{args.seed}",
            config=vars(args)
        )
    
    output = transfer_knowledge(args, dataset, logger)
    
    # Summary
    result = output['result']
    if 'hm' in result:
        logger.info(f"\nFinal: {dataset}: reg={result['reg_coef']}, Base={result['base_acc']:.2f}%, Novel={result['novel_acc']:.2f}%, HM={result['hm']:.2f}%")
    else:
        logger.info(f"\nFinal: {dataset}: reg={result['reg_coef']}, Acc={result['accuracy']:.2f}%")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
