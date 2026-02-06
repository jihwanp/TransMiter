import os
import argparse

import torch
import ast

def str2list(s):
    # import pdb;pdb.set_trace()
    print(s)
    print(s[0])
    v = ast.literal_eval(s.replace(" ", ""))
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=None,
        help='Directory for caching models from OpenCLIP'
    )
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for the step scheduler')
    
    parser.add_argument('--model_arch', type=str, default='ViT-B-32', help='Model architecture')
    
    parser.add_argument('--eval', action='store_true', help='Evaluate naive model')
    parser.add_argument('--experiment_name', type=str, default='debug', help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='Cars', help='Dataset',choices=['Cars','Food101', 'EuroSAT', 'GTSRB', 'MNIST', 'SUN397', 'SVHN', 'DTD','RESISC45','Caltech','Aircraft','Pets','UCF','ImageNet','Flowers','ImageNet_a','ImageNet_r','ImageNet_sketch','ImageNetv2'])
    
    # proxy model
    parser.add_argument('--use_train_for_proxy', action='store_true', help='Use test set for the proxy model')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='Data ratio for the proxy model')
    parser.add_argument('--ldn_network', type=str, default='LogitAlone', choices=['LogitAlone'], help='Logit delta network')
                        
    parser.add_argument('--alpha_scale', type=float, default=1.0, help='Scale factor for the logit delta network') 
    parser.add_argument('--num_fc_layers', type=int, default=1, help='Number of fully connected layers in the logit delta network')
    parser.add_argument('--bottleneck_dim', type=int, default=512, help='Number of dimensions in the logit delta network')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function in the logit delta network')
    parser.add_argument('--use_train_preprocess', action='store_true', help='Use train preprocess for the logit delta network')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the logit delta network')
    parser.add_argument('--logit_stand', action='store_true', help='Standardize the logits before passing to the logit delta network')
    parser.add_argument('--source_model', type=str, default='ViT-B-16', help='Source model for the logit delta network')
    parser.add_argument('--source_model_path', type=str, default='./checkpoints/ViT-B-16', help='Path to the source model')
    parser.add_argument('--target_model', type=str, default='ViT-L-14', help='Target model for the logit delta network')
    parser.add_argument('--target_model_path', type=str, default='./checkpoints/ViT-L-14', help='Path to the target model')
    parser.add_argument('--eval_source_and_target', action='store_true', help='Evaluate the source and target models')
    
    parser.add_argument('--eval_with_prelogits', action='store_true', help='Evaluate the proxy model with prelogits')
    
    # text encoder
    parser.add_argument('--text_encoder', type=str, default='roberta', help='Text encoder')
    parser.add_argument('--text_feature_dim', type=int, default=1024, help='Feature dimension of the text encoder')
    
    # PDN arguments
    parser.add_argument('--share_layer', action='store_true', help='Share the layers in the logit delta network')
    parser.add_argument('--logit_norm_strategy', type=str, default='standard', choices=['softmax','standard','none','prestat','center','prestat_bn','scale','l2'], help='Logit normalization strategy to the logit delta network')
    parser.add_argument('--use_aux_loss', action='store_true', help='Use auxiliary loss for the logit delta network')
    
    # ver2 (common)
    parser.add_argument('--transfer_direct_pred', action='store_true', help='Directly predict the logits')
    parser.add_argument('--transfer_dropout', type=float, default=0.0, help='Dropout for the transfer learning')
    parser.add_argument('--transfer_fc_layers', type=int, default=1, help='Number of fully connected layers for transfer learning')
    parser.add_argument('--transfer_num_refine_layers', type=int, default=1, help='Number of layers to refine for transfer learning')
    parser.add_argument('--use_orthogonality', action='store_true', help='Use orthogonality for the transfer learning')  
    parser.add_argument('--mul_dim', type=int, default=-1, help='Multiplier for the dimension of the transfer learning')
    parser.add_argument('--proj_before_pullback', action='store_true', help='Add projection before pullback for the transfer learning')
    parser.add_argument('--mul_std_transfer', action='store_true', help='Multiply the standard deviation for the transfer learning')
    parser.add_argument('--input_scale', type=float, default=1.0, help='Input scale for the transfer learning')
    parser.add_argument('--transfer_procrutes', action='store_true', help='Use procrutes for transfer')
    parser.add_argument('--proj_dim', type=int, default=128, help='Projection dimension for the transfer learning')
    parser.add_argument('--use_skip_connection', action='store_true', help='Use skip connection for the transfer learning')
    parser.add_argument('--temperature_strategy', type=str, default='none', choices=['none','linear','exp'], help='Temperature scaling strategy for the transfer learning')
    parser.add_argument('--num_aux', type=int, default=0, help='Number of auxiliary for the transfer learning')
    parser.add_argument('--use_interpolated_input', action='store_true', help='Use interpolated input for the transfer learning')
    parser.add_argument('--max_interpolation', type=float, default=0.0, help='Maximum interpolation for the transfer learning')
    parser.add_argument('--sep_proj', action='store_true', help='Separate projection for the transfer learning')
    parser.add_argument('--use_proj_ln', action='store_true', help='Use layer norm for the projection for the transfer learning')
    parser.add_argument('--input_dropout', type=float, default=0.0, help='Input dropout for the transfer learning')
    parser.add_argument('--use_class_embed', action='store_true', help='Use class embedding for the transfer learning')
    parser.add_argument('--init_all', action='store_true', help='Initialize all the layers for the transfer learning')
    parser.add_argument('--pad_classes', action='store_true', help='Pad the classes for the transfer learning')
    parser.add_argument('--class_padding_strategy', type=str, default='none', choices=['none','random','wordnet','wordnet_hard','multi_prompt','description','openimage'], help='Class padding strategy for the transfer learning')
    parser.add_argument('--aux_pad_coef', type=float, default=1.0, help='Auxiliary padding coefficient for the transfer learning')
    parser.add_argument('--word_embedding_dir', type=str, default='./word_embed/', help='Word embedding directory for the transfer learning')
    parser.add_argument('--class_sampling_strategy', type=str, default='random', choices=['random','fps','synonym','mix_fps','easy_fps','hard_neg','uniform','all','top_std','std_fps'], help='Class sampling strategy for the transfer learning')
    parser.add_argument('--use_mean_variance_loss', action='store_true', help='Use mean variance loss for the transfer learning')
    parser.add_argument('--no_ln', action='store_true', help='Do not use layer norm for the transfer learning')
    parser.add_argument('--padding_template', type=str, default='photo', help='Template for the class padding strategy')
    parser.add_argument('--only_task_class_transfer', action='store_true', help='Only transfer the task classes')
    parser.add_argument('--tot_logit_dim', type=int, default=-1, help='Total logit dimension for the transfer learning')
    parser.add_argument('--procrutes_norm_strategy', type=str, default='none', choices=['none','standard','center','norm'], help='Procrutes normalization strategy for the transfer learning')
    parser.add_argument('--rotation', action='store_true', help='Rotation orthonormal matrix for the transfer learning')
    parser.add_argument('--exponential_mapping', action='store_true', help='Exponential mapping between basis matrices for the transfer learning')
    parser.add_argument('--target_interpolate_ratio', type=float, default=0.5, help='Interpolation ratio for the target logits')
    parser.add_argument('--use_l2_norm_features', action='store_true', help='Use l2 normalized features for the transfer learning')
    parser.add_argument('--direct_procrutes', action='store_true', help='Direct procrutes for the transfer learning')
    parser.add_argument('--truncate_ratio', type=float, default=-1, help='Truncate ratio for the transfer learning')
    parser.add_argument('--use_all_logits_for_loss', action='store_true', help='Use all the logits for the loss')
    parser.add_argument('--latent_procrutes', action='store_true', help='Latent procrutes for the transfer learning')
    parser.add_argument('--pseudo_align', action='store_true', help='Pseudo align for the transfer learning')
    parser.add_argument('--noise_alpha', type=float, default=0.0, help='Noise alpha for the transfer learning')
    parser.add_argument('--reg_procrutes_coef', type=float, default=0.0, help='Regularization coefficient for the procrutes')
    parser.add_argument('--class_wise_noise', action='store_true', help='Class wise noise for the transfer learning')
    parser.add_argument('--use_dtemp_for_neg', action='store_true', help='Use data specific templates for the negative classes')
    parser.add_argument('--aux_templates', type=str, default=None, help='Auxiliary templates for the transfer learning')
    parser.add_argument('--extract_latent_place',type=str,default='none', choices = ['none','post_mlp','post_logits','both_feats','both_feats_v2'],help='Place to extract latent')
    parser.add_argument('--intermediate_models',type=str2list,default=[],help='Intermediate models for the transfer learning')
    
    parser.add_argument('--hra_r',type=int,default=None, help='rank of householder reflection adaptation')
    parser.add_argument('--only_input_latent_transfer', action='store_true', help='Only input latent transfer')
    
    # for post ensemble model
    parser.add_argument('--ensemble_source_model_list',type=str2list,default=[],help='Ensemble source model list for the transfer learning')
    
    parser.add_argument('--init_template',type=str,default=None,help='Template initialization')
    
    parser.add_argument('--transfer_ensemble', action='store_true', help='Ensemble transfer learning')
    
    parser.add_argument('--train_with_labels', action='store_true', help='Train with labels for the transfer learning')    
    parser.add_argument('--post_train_with_labels', action='store_true', help='Post train with labels for the transfer learning')
    parser.add_argument('--post_train_online_extraction', action='store_true', help='Post train online extraction for the transfer learning')
    parser.add_argument('--keep_orthogonality_post_train', action='store_true', help='Keep orthogonality post train for the freeze_projection')
    parser.add_argument('--freeze_projection', action='store_true', help='Freeze projection for the transfer learning')
    parser.add_argument('--skip_knowledge_extraction', action='store_true', help='Skip knowledge extraction for the transfer learning')
    parser.add_argument('--load_transferred_model', action='store_true', help='Load pretrained transferred model')
    parser.add_argument('--load_post_trained_model', action='store_true', help='Load post trained model. only for evaluation (expecially ensemble)')
    parser.add_argument('--only_extract_knowledge', action='store_true', help='Only extract knowledge for the transfer learning')
    
    parser.add_argument('--pretrained_pa_dir', type=str, default='', help='Pretrained PA directory for the transfer learning')
    
    parser.add_argument('--use_precompute_features', action='store_true', help='Use precomputed features for the transfer learning')
    parser.add_argument('--feature_extract', action='store_true', help='Extract features for the transfer learning')
    parser.add_argument('--ft_strategy',type=str2list, default=['fullfinetune'], help='Fine tuning strategy for the transfer learning')
    parser.add_argument('--loader_ver2', action='store_true', help='Use the new loader')
    
    # ema
    parser.add_argument('--use_ema', action='store_true', help='Use EMA for the transfer learning')
    parser.add_argument('--ema_decay', type=float, default=0.9998, help='EMA decay for the transfer learning')
    
    # baselines
    parser.add_argument('--baseline', type=str, default='none', choices=['none','proxy','base_change','ensemble'], help='Baseline method')
    parser.add_argument('--baseline_alpha', type=float, default=1.0, help='Alpha for the baseline method')
    
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='proxy_vlm', help='Wandb project name')
    parser.add_argument('--wandb_group', type=str, default='debug', help='Wandb group name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--use_class_wise_stats', action='store_true', help='Calculate class wise statistics')
    parser.add_argument('--post_normalize', action='store_true', help='Post normalize the logits')
    
    parser.add_argument('--no_normalize_target', action='store_true', help='Do not normalize the target logits')
    parser.add_argument('--use_source_ft_stats', action='store_true', help='Use source fine-tuned stats')
    
    # ensemble (250627)
    parser.add_argument('--ensemble_strategy', type=str, default='average', choices=['average','weighted','learn_scale','moe_logits','moe_input_features','moe_output_features'], help='Ensemble strategy')
    
    parser.add_argument('--eval_all', action='store_true', help='Evaluate all the models')
    parser.add_argument('--few_shot', action='store_true', help='Few shot learning')
    parser.add_argument('--base2novel', action='store_true', help='Base to novel few shot learning')
    parser.add_argument('--setting', type=str, default=None, choices=['few_shot','base2novel','cross_data','dg'], help='Settings for few shot learning')
    parser.add_argument('--num_shot', type=int, default=16, choices=[1,2,4,8,16],help='Number of ways for few shot learning')
    parser.add_argument('--use_all_train', action='store_true', help='Use all the training data for few shot learning')
    parser.add_argument('--log_exclude',type=str2list, default=[], help='Exclude the logs')
    
    parser.add_argument('--dg', action='store_true', help='Domain generalization')
    parser.add_argument('--dg_dataset', type=str, default='imagenet_a', help='Domain generalization dataset', choices=['imagenet_a','imagenet_r','imagenet_sketch','imagenetv2'])
    
    parser.add_argument('--use_ft_logits', action='store_true', help='Use fine-tuned logits in dataloader')
    # iccv
    parser.add_argument('--seperate_ensemble', action='store_true', help='Seperate ensemble')
    parser.add_argument('--loader_ensemble', action='store_true', help='Use ensemble loader')
    
    parser.add_argument('--alpha_tuning_after_procrutes', action='store_true', help='Alpha tuning after procrutes')
    parser.add_argument('--apply_tuned_alpha_after_procrutes', action='store_true', help='Apply tuned alpha after procrutes')
    parser.add_argument('--apply_tuned_alpha_after_post_train', action='store_true', help='Apply tuned alpha after post train')
    
    parser.add_argument('--eval_correlation', action='store_true', help='Evaluate correlation')
    parser.add_argument('--visualize_correct_samples', action='store_true', help='Visualize correct samples')
    parser.add_argument('--eval_adapter_ensemble', action='store_true', help='Evaluate adapter ensemble')
    parser.add_argument('--adapter_ensemble_beta', type=float, default=0.5, help='Beta for adapter ensemble')
    
    parser.add_argument('--apply_tf_ensemble', action='store_true', help='Apply training-free ensemble')
    parser.add_argument('--self_ensemble_alpha', type=float, default=0.5, help='Alpha for self ensemble (zero-shot model)')
    
    parser.add_argument('--ensemble_multi_proxy', action='store_true', help='Ensemble multi proxy')
    
    parser.add_argument('--use_transmiter_source_model', action='store_true', help='Use transmiter source model')
    parser.add_argument('--source_transmiter_model_path', type=str, default='./checkpoints/ViT-B-16->ViT-L-14/reg_50/', help='Path to the source transmiter model')
    
    return parser


def parse_arguments(parser=None):
    """Parse arguments. Can optionally extend a base parser."""
    if parser is None:
        parser = get_base_parser()
    
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    
    if parsed_args.extract_latent_place in ['post_mlp','post_logits']:
        assert parsed_args.only_input_latent_transfer, f'Only input latent transfer is only possible when latent transfer it done in {parsed_args.extract_latent_place}'
        
    return parsed_args
