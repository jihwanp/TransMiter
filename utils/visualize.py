import os
import json
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
import torch.nn.functional as F

def visualize_correct_samples(source_zs_model, source_ft_model, target_model, proxy_model, dataset_name, seed_log_path, args, pre_transfer=False):
    """
    Visualize the comparison of model predictions before and after training.
    
    Groups samples based on prediction status:
    'xxx': source_zs wrong, source_ft wrong, target_zs wrong
    'xxo': source_zs wrong, source_ft wrong, target_zs correct
    'xox': source_zs wrong, source_ft correct, target_zs wrong
    'xoo': source_zs wrong, source_ft correct, target_zs correct
    'oxx': source_zs correct, source_ft wrong, target_zs wrong
    'oxo': source_zs correct, source_ft wrong, target_zs correct
    'oox': source_zs correct, source_ft correct, target_zs wrong
    'ooo': source_zs correct, source_ft correct, target_zs correct
    
    For each group, show how many samples the proxy model gets correct/wrong.
    """
    device = args.device
    source_zs_model.eval()
    target_model.eval()
    proxy_model.eval()
    if isinstance(source_ft_model, list):
        for m in source_ft_model:
            m.eval()
    else:
        source_ft_model.eval()

    dataset = get_dataset(
        dataset_name,
        proxy_model.val_preprocess,
        location=args.data_location,
        batch_size=128,
        args=args
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

    # Initialize group counters and proxy model correct/wrong counts per group
    groups = ['xxx', 'xxo', 'xox', 'xoo', 'oxx', 'oxo', 'oox', 'ooo']
    group_counts = {g: 0 for g in groups}
    proxy_correct_counts = {g: 0 for g in groups}
    
    # For base2novel setting
    if hasattr(dataset, 'base_class_idx'):
        base_group_counts = {g: 0 for g in groups}
        base_proxy_correct_counts = {g: 0 for g in groups}
        novel_group_counts = {g: 0 for g in groups}
        novel_proxy_correct_counts = {g: 0 for g in groups}

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            y = data['labels'].to(device)
            
            # Get source zero-shot model predictions
            source_zs_inputs = data[f'{source_zs_model.image_encoder.model_name}_features'].to(device)
            source_zs_inputs /= source_zs_inputs.norm(dim=-1, keepdim=True)
            source_zs_logits = source_zs_model.classification_head(source_zs_inputs)[:,:args.num_classes]
            
            all_source_ft_logits = []
            if args.use_ft_logits:
                source_ft_inputs = data[f'{source_ft_model[0].image_encoder.model_name}_ft_logits']
                for ft_mod, ft_inp in zip(source_ft_model, source_ft_inputs):
                    source_ft_logits = ft_inp.to(device)
                    all_source_ft_logits.append(source_ft_logits[:,:args.num_classes])
                source_ft_logits = torch.stack(all_source_ft_logits).mean(dim=0)
            else:
                # Get source finetuned predictions
                source_ft_inputs = data[f'{source_ft_model[0].image_encoder.model_name}_ft_features']
                for ft_mod, ft_inp in zip(source_ft_model, source_ft_inputs):
                    ft_inp = ft_inp.to(device)
                    ft_inp /= ft_inp.norm(dim=-1, keepdim=True)
                    source_ft_logits = ft_mod.classification_head(ft_inp)[:,:args.num_classes]
                    all_source_ft_logits.append(source_ft_logits)
                source_ft_logits = torch.stack(all_source_ft_logits).mean(dim=0)
            
            # Get target model predictions
            target_zs_inputs = data[f'{target_model.image_encoder.model_name}_features'].to(device)
            target_zs_inputs /= target_zs_inputs.norm(dim=-1, keepdim=True)
            target_zs_logits = target_model.classification_head(target_zs_inputs)
            
            # Get proxy model predictions
            target_proxy_logits = proxy_model(None, target_zs_logits)['logits'][:,:args.num_classes]
            
            target_zs_logits = target_zs_logits[:,:args.num_classes]
            
            # print('source_zs_logits',source_zs_logits.shape)
            # print('source_ft_logits',source_ft_logits.shape)
            # print('target_zs_logits',target_zs_logits.shape)
            # print('target_proxy_logits',target_proxy_logits.shape)
            
            # Update group counts
            for idx in range(len(y)):
                label = y[idx].item()
                
                # Determine predictions based on base2novel setting or regular setting
                if hasattr(dataset, 'base_class_idx'):
                    is_base = label in dataset.base_class_idx
                    
                    if is_base:
                        # For base classes, restrict predictions to base class indices
                        # Create mask for non-base classes
                        mask = torch.ones_like(source_zs_logits[idx]).bool()
                        for base_idx in dataset.base_class_idx:
                            mask[base_idx] = False
                        
                        # Apply mask by setting non-base logits to negative infinity
                        masked_source_zs_logits = source_zs_logits[idx].clone()
                        masked_source_zs_logits[mask] = float('-inf')
                        source_zs_pred = masked_source_zs_logits.argmax().item()
                        
                        masked_source_ft_logits = source_ft_logits[idx].clone()
                        masked_source_ft_logits[mask] = float('-inf')
                        source_ft_pred = masked_source_ft_logits.argmax().item()
                        
                        masked_target_zs_logits = target_zs_logits[idx].clone()
                        masked_target_zs_logits[mask] = float('-inf')
                        target_zs_pred = masked_target_zs_logits.argmax().item()
                        
                        masked_target_proxy_logits = target_proxy_logits[idx].clone()
                        masked_target_proxy_logits[mask] = float('-inf')
                        target_proxy_pred = masked_target_proxy_logits.argmax().item()
                    else:
                        # For novel classes, restrict predictions to novel class indices
                        # Create mask for non-novel classes
                        mask = torch.ones_like(source_zs_logits[idx]).bool()
                        for novel_idx in dataset.novel_class_idx:
                            mask[novel_idx] = False
                            
                        # Apply mask by setting non-novel logits to negative infinity
                        masked_source_zs_logits = source_zs_logits[idx].clone()
                        masked_source_zs_logits[mask] = float('-inf')
                        source_zs_pred = masked_source_zs_logits.argmax().item()
                        
                        masked_source_ft_logits = source_ft_logits[idx].clone()
                        masked_source_ft_logits[mask] = float('-inf')
                        source_ft_pred = masked_source_ft_logits.argmax().item()
                        
                        masked_target_zs_logits = target_zs_logits[idx].clone()
                        masked_target_zs_logits[mask] = float('-inf')
                        target_zs_pred = masked_target_zs_logits.argmax().item()
                        
                        masked_target_proxy_logits = target_proxy_logits[idx].clone()
                        masked_target_proxy_logits[mask] = float('-inf')
                        target_proxy_pred = masked_target_proxy_logits.argmax().item()
                else:
                    # Regular setting: standard argmax across all classes
                    source_zs_pred = source_zs_logits[idx].argmax().item()
                    source_ft_pred = source_ft_logits[idx].argmax().item()
                    target_zs_pred = target_zs_logits[idx].argmax().item()
                    target_proxy_pred = target_proxy_logits[idx].argmax().item()
                
                # Determine correctness
                source_zs_correct = source_zs_pred == label
                source_ft_correct = source_ft_pred == label
                target_zs_correct = target_zs_pred == label
                target_proxy_correct = target_proxy_pred == label
                
                # Create group key and update counts
                group_key = f"{'o' if source_zs_correct else 'x'}{'o' if source_ft_correct else 'x'}{'o' if target_zs_correct else 'x'}"
                
                # Update overall group counts
                group_counts[group_key] += 1
                if target_proxy_correct:
                    proxy_correct_counts[group_key] += 1
                
                # For base2novel setting
                if hasattr(dataset, 'base_class_idx'):
                    if is_base:
                        base_group_counts[group_key] += 1
                        if target_proxy_correct:
                            base_proxy_correct_counts[group_key] += 1
                    else:
                        novel_group_counts[group_key] += 1
                        if target_proxy_correct:
                            novel_proxy_correct_counts[group_key] += 1

    if pre_transfer:
        add_name = 'pre_transfer'
    else:
        add_name = 'post_transfer'
    # Create plots
    if hasattr(dataset, 'base_class_idx'):
        # Create separate plots for base and novel classes
        create_prediction_group_plot(base_group_counts, base_proxy_correct_counts, 
                                    os.path.join(seed_log_path, f'prediction_groups_base_{dataset_name}_{add_name}.png'),
                                    f'Target+Proxy Model Prediction Analysis - Base Classes ({dataset_name})')
        
        create_prediction_group_plot(novel_group_counts, novel_proxy_correct_counts, 
                                    os.path.join(seed_log_path, f'prediction_groups_novel_{dataset_name}_{add_name}.png'),
                                    f'Target+Proxy Model Prediction Analysis - Novel Classes ({dataset_name})')
    else:
        # Create a single plot for all classes
        create_prediction_group_plot(group_counts, proxy_correct_counts, 
                                   os.path.join(seed_log_path, f'prediction_groups_{dataset_name}_{add_name}.png'),
                                   f'Target+Proxy Model Prediction Analysis ({dataset_name})')
        
    # Print statistics
    print(f"\nDataset: {dataset_name}")
    print("\nGroup Statistics (Source ZS / Source FT / Target ZS):")
    for group in groups:
        print(f"{group}: {group_counts[group]} samples, {proxy_correct_counts[group]} correct by proxy ({proxy_correct_counts[group]/max(1, group_counts[group]):.2%})")
        
    if hasattr(dataset, 'base_class_idx'):
        print("\nBase Classes:")
        for group in groups:
            print(f"{group}: {base_group_counts[group]} samples, {base_proxy_correct_counts[group]} correct by proxy ({base_proxy_correct_counts[group]/max(1, base_group_counts[group]):.2%})")
        
        print("\nNovel Classes:")
        for group in groups:
            print(f"{group}: {novel_group_counts[group]} samples, {novel_proxy_correct_counts[group]} correct by proxy ({novel_proxy_correct_counts[group]/max(1, novel_group_counts[group]):.2%})")


def create_prediction_group_plot(group_counts, proxy_correct_counts, save_path, title):
    """
    Create a bar plot showing correct/incorrect predictions by the proxy model for different groups.
    
    Args:
        group_counts: Dictionary with counts of samples in each group
        proxy_correct_counts: Dictionary with counts of correctly predicted samples by proxy model
        save_path: Path to save the figure
        title: Title for the plot
    """
    groups = ['xxx', 'xxo', 'xox', 'xoo', 'oxx', 'oxo', 'oox', 'ooo']
    
    # Calculate proxy incorrect counts
    proxy_incorrect_counts = {g: group_counts[g] - proxy_correct_counts[g] for g in groups}
    
    # Data for plotting
    group_labels = groups
    total_heights = [group_counts[g] for g in groups]
    correct_heights = [proxy_correct_counts[g] for g in groups]
    incorrect_heights = [proxy_incorrect_counts[g] for g in groups]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot stacked bars
    bars = plt.bar(group_labels, total_heights, color='red')
    
    # Plot correct predictions (blue part of the bars)
    plt.bar(group_labels, correct_heights, color='blue')
    
    # Add counts as text on the bars
    for i, (total, correct) in enumerate(zip(total_heights, correct_heights)):
        if total > 0:
            # Add total count on top of the bar
            plt.text(i, total + max(total_heights)*0.02, str(total), 
                     ha='center', va='bottom', fontweight='bold')
            
            # Add correct count in the middle of the blue part if there's enough space
            if correct > max(total_heights)*0.05:
                plt.text(i, correct/2, str(correct), 
                        ha='center', va='center', color='white', fontweight='bold')
            
            # Add correct percentage
            percentage = correct / total * 100
            plt.text(i, total / 2, f"{percentage:.1f}%", 
                     ha='center', va='center', fontweight='bold', 
                     color='white' if percentage > 30 else 'black')
    
    # Set labels and title
    plt.xlabel('Prediction Groups (Source ZS / Source FT / Target ZS)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Target+Proxy Correct'),
        Patch(facecolor='red', label='Target+Proxy Incorrect')
    ]
    plt.legend(handles=legend_elements, fontsize=10)
    
    # Add explanation of group labels
    plt.figtext(0.5, 0.01, 
                "Group labels: 'o' means correct, 'x' means wrong\n"
                "xxx: all models wrong, ooo: all models correct", 
                ha='center', fontsize=10)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path)
    plt.close()