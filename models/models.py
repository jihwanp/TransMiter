import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.logit_stand import logit_stand

import geotorch

activation_dicts = {"relu":nn.ReLU, 'silu':nn.SiLU,'gelu':nn.GELU, 'leaky_relu':nn.LeakyReLU}

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LogitAdjustHead_ver3(nn.Module):
    def __init__(self, args):
        super(LogitAdjustHead_ver3, self).__init__()
        self.class_embeds = None
        
        self.proj_dim = args.proj_dim
        
        self.use_orthogonality = args.use_orthogonality
        
        self.num_classes = args.num_classes
        self.proj_before_pullback = args.proj_before_pullback
        
        self.class_padding_strategy = args.class_padding_strategy
        if self.class_padding_strategy!='none':
            
            self.num_classes = self.proj_dim if args.tot_logit_dim ==-1 else args.tot_logit_dim
            self.num_pad = self.num_classes - args.num_classes
        else:
            self.num_pad = 0
            
        self.mul_std_transfer = args.mul_std_transfer
        self.input_dropout = args.input_dropout
        self.use_class_embed = args.use_class_embed
        if self.use_class_embed:
            self.class_embed_proj = nn.Linear(args.text_feature_dim, self.proj_dim)
        
        if self.input_dropout>0.:
            assert args.logit_norm_strategy == 'softmax', "Input dropout only support softmax normalization"
        if args.mul_dim!=-1:
            self.bottleneck_dim = args.mul_dim * self.proj_dim
        else:
            self.bottleneck_dim = args.bottleneck_dim
        self.scale_factor = args.alpha_scale        
        self.sep_proj = args.sep_proj
        self.only_task_class_transfer = args.only_task_class_transfer
        
        self.hra_r = args.hra_r
        if self.hra_r is not None:
            self.in_hra_u = nn.Parameter(torch.randn(self.proj_dim, self.hra_r))
        else:
            
            if self.only_task_class_transfer:
                assert self.class_padding_strategy != 'none', 'class_padding_strategy should be not none'
                self.in_proj = nn.Linear(args.num_classes, args.num_classes, bias=False)
                
                self.neg_proj = torch.eye(self.num_pad,device=self.in_proj.weight.device,dtype= self.in_proj.weight.dtype)
                
            else:
                self.in_proj = nn.Linear(self.num_classes, self.proj_dim, bias=False)
        
        if self.use_orthogonality:
            
            assert self.proj_dim >= self.num_classes, f"proj dim ({self.proj_dim}) should be larger or equal than num_classes ({self.num_classes}) for orthogonality"
            
        if args.no_ln:
            mlp_layer = nn.Sequential(
                nn.Linear(self.proj_dim, self.bottleneck_dim),
                activation_dicts[args.activation](),
                nn.Dropout(args.transfer_dropout),
                nn.Linear(self.bottleneck_dim, self.proj_dim),
            )
        else:
            mlp_layer = nn.Sequential(
                nn.LayerNorm(self.proj_dim),
                nn.Linear(self.proj_dim, self.bottleneck_dim),
                activation_dicts[args.activation](),
                nn.Dropout(args.transfer_dropout),
                nn.Linear(self.bottleneck_dim, self.proj_dim),
            )        
        
        self.mlp_layers = nn.ModuleList([mlp_layer for _ in range(args.transfer_fc_layers)]) if args.share_layer else _get_clones(mlp_layer, args.transfer_fc_layers)
        if self.proj_before_pullback:
            self.proj_bp = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        
        self.use_proj_ln = args.use_proj_ln
        if self.use_proj_ln:
            self.proj_ln = nn.LayerNorm(self.proj_dim)
        
        if self.sep_proj:
            if self.hra_r is None:
                self.out_proj = nn.Linear(self.proj_dim, self.num_classes,bias=False)
            else:
                self.out_hra_u = nn.Parameter(torch.randn(self.proj_dim, self.hra_r))
        
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
                
        self.transfer_direct_pred = args.transfer_direct_pred
        self.logit_norm_strategy = args.logit_norm_strategy
        
        if self.use_orthogonality:
            if self.hra_r is not None:
                self.in_hra_u.data
                if self.sep_proj:
                    self.out_hra_u.data = self.in_hra_u.data
            else:
                geotorch.orthogonal(self.in_proj, "weight")
                if self.sep_proj:
                    geotorch.orthogonal(self.out_proj, "weight")
                    self.out_proj.weight = self.in_proj.weight.T.clone()

            for l in self.mlp_layers:
                l[-1].weight.data.zero_()
                l[-1].bias.data.zero_()
        else:
            self.in_proj.weight.data.zero_()
            torch.nn.init.eye_(self.in_proj.weight)
            for l in self.mlp_layers:
                l[-1].weight.data.zero_()
                l[-1].bias.data.zero_()
            if self.sep_proj:
                self.out_proj.weight.data.zero_()
                torch.nn.init.eye_(self.out_proj.weight)
        
        self.transfer_mode = False
        self.transfer_procrutes_mode = False
        
        self.transfer_out_proj = None
        
        self.use_class_wise_stats = args.use_class_wise_stats
        self.pre_compute_stats = None
        
        self.input_scale = args.input_scale
        self.use_skip_connection = args.use_skip_connection
        self.use_aux_loss = args.use_aux_loss
        
        self.num_aux = args.num_aux
        
        if args.init_all:
            for p in self.parameters():
                if len(p.shape) > 1:
                    nn.init.xavier_uniform_(p)

        if self.logit_norm_strategy=='l2':
            assert self.transfer_direct_pred, "L2 normalization only support direct prediction"
        self.use_l2_norm_features = args.use_l2_norm_features

        if self.hra_r is not None:
            self.in_proj = nn.Linear(self.num_classes, self.proj_dim, bias=False)
            self.in_proj.requires_grad_(False)
            if self.sep_proj:
                self.out_proj = nn.Linear(self.proj_dim, self.num_classes,bias=False)
                self.out_proj.requires_grad_(False)
            self.transfer_phase = False
            
        self.extract_latent_place = args.extract_latent_place
        if self.extract_latent_place !='none':
            if 'both_feats' in self.extract_latent_place:
                self.latent_mapper_in = None
                self.latent_mapper_out = None
            else:
                self.latent_mapper = None
        
        # for orthogonal post training
        self.orth_mapper = None
        
    def hra_projection(self,place = 'push'):
        if place=='push':
            hra_u = self.in_hra_u
        elif place=='pull':
            hra_u = self.out_hra_u if self.sep_proj else self.in_hra_u
        else:
            raise NotImplementedError("Place not implemented")
        
        hra_u_norm = hra_u/hra_u.norm(dim=0,keepdim=True)
    
        weight = torch.eye(self.proj_dim,device=hra_u_norm.device,dtype=hra_u_norm.dtype)
        for i in range(self.hra_r):
            ui = hra_u_norm[:, i].view(-1, 1)
            weight = weight - 2*weight@ui@ui.T
        truncated_weight = weight[:self.num_classes]
        if place =='push':
            self.in_proj.weight.data = truncated_weight.T.data
        elif place=='pull':
            if self.sep_proj:
                self.out_proj.weight.data = truncated_weight.data
        else:
            raise NotImplementedError("Place not implemented")
        return truncated_weight
    
    def transform_logits(self,logits):
        logit_mean, logit_std = None, None
            
        if self.pre_compute_stats is not None:
            if self.use_class_wise_stats:
                logit_mean = self.pre_compute_stats['class_wise_logit_mean'][None]
                logit_std = self.pre_compute_stats['class_wise_logit_std'][None]
            else:
                logit_mean = self.pre_compute_stats['all_logit_mean']
                logit_std = self.pre_compute_stats['all_logit_std']
            stand_logit = (logits - logit_mean) / logit_std
        else:
            if self.logit_norm_strategy == 'softmax':
                if self.input_dropout>0. and self.training:
                    drop_prob = 1-logits.softmax(dim=-1)
                    indices = torch.multinomial(drop_prob, int(self.num_classes*self.input_dropout), replacement=False)
                    logits = torch.scatter(logits,-1, indices, -float('inf'))
                stand_logit = logits.softmax(dim=-1)
            elif self.logit_norm_strategy == 'standard':
                stand_logit,logit_mean,logit_std = logit_stand(logits,True)
                stand_logit = stand_logit/self.input_scale
            elif self.logit_norm_strategy == 'center':
                logit_mean = logits.mean(dim=-1, keepdim=True)
                stand_logit = logits - logit_mean
                stand_logit/= self.input_scale
            elif self.logit_norm_strategy == 'scale':
                stand_logit = logits / 100.
            elif self.logit_norm_strategy == 'l2':
                stand_logit = F.normalize(logits, p=2, dim=-1)
            else:
                stand_logit = logits
        return stand_logit, logit_mean,logit_std
    
    def push_latent(self,stand_logit):
        if self.use_orthogonality:
            if self.only_task_class_transfer:
                in_proj = torch.block_diag(self.in_proj.weight, self.neg_proj.to(device=stand_logit.device))
                in_proj = in_proj.T
            else:
                if self.hra_r is not None:
                    in_proj = self.hra_projection('push') if not self.transfer_phase else self.in_proj.weight.T
                else:
                    in_proj = self.in_proj.weight.T
            x = stand_logit@in_proj
            
            if self.in_proj.bias is not None:
                in_bias = self.in_proj.bias
                x = x + in_bias
                
        else:
            in_proj = self.in_proj.weight.T
            x = stand_logit@in_proj
        return x
    
    def pull_latent(self, x):
        if self.use_orthogonality:
            if self.in_proj.bias is not None:
                in_bias = self.in_proj.bias
                x = x - in_bias
            if self.only_task_class_transfer:
                out_proj = torch.block_diag(self.in_proj.weight, self.neg_proj.to(device=x.device)) if self.transfer_out_proj is None \
                    else torch.block_diag(self.transfer_out_proj.weight, self.neg_proj.to(device=x.device))
                out_proj = out_proj
            else:
                if self.hra_r is not None:
                    out_proj = self.hra_projection('pull').T if self.transfer_out_proj is None else self.transfer_out_proj.weight
                else:
                    if self.sep_proj:
                        out_proj = self.out_proj.weight.T if self.transfer_out_proj is None else self.transfer_out_proj.weight
                    else:
                        out_proj = self.in_proj.weight if self.transfer_out_proj is None else self.transfer_out_proj.weight
            z = x@out_proj
        else:
            if self.sep_proj:
                out_proj = self.out_proj.weight.T if self.transfer_out_proj is None else self.transfer_out_proj.weight
                z = x@out_proj
            else:
                out_proj = self.in_proj.weight if self.transfer_out_proj is None else self.transfer_out_proj.weight
                z = x@out_proj
        return z
    
    def transform_back_logits(self, orig_logit, stand_logit, z, logit_mean, logit_std, alpha):
        if self.transfer_direct_pred:
            if self.logit_norm_strategy=='standard':
                z = z * logit_std*self.input_scale + logit_mean
            elif self.logit_norm_strategy=='center':
                z = z * self.input_scale + logit_mean
            elif self.pre_compute_stats is not None:
                
                z = z * logit_std + logit_mean
            elif self.logit_norm_strategy == 'softmax':
                z = F.log_softmax(z, dim=-1)
            elif self.logit_norm_strategy == 'scale':
                z = z*100.
            else:
                z = z
        
            return z
            
        else:
            if self.logit_norm_strategy == 'softmax':
                
                log_prob = F.log_softmax(orig_logit, dim=-1)
                return log_prob + alpha*z
            elif self.logit_norm_strategy == 'standard':
                z = z * logit_std* self.input_scale
                return z + orig_logit
            elif self.logit_norm_strategy == 'center':
                z = z * self.input_scale + logit_mean
                return orig_logit + alpha*z
            elif self.pre_compute_stats is not None:
                z = z * logit_std
                return orig_logit + alpha*z
            elif self.logit_norm_strategy == 'scale':
                refine_logit = stand_logit + alpha*z
                return refine_logit * 100
                
            else:
                return orig_logit + alpha*z

    def forward(self, logits):
        logit_dicts = {'aux_logits':[]}
        alpha = self.scale_factor
        
        stand_logit, logit_mean, logit_std = self.transform_logits(logits)
        
        # PRE LOGITS
        if self.transfer_procrutes_mode and self.extract_latent_place=='none':
            return stand_logit

        if self.use_class_embed:
            x = self.class_embed_proj(self.class_embeds)
            x = topk_scores.unsqueeze(-1)*x[topk_indices]
            x = x.mean(dim=1)
            
        else:
            # PRE MLP
            x = self.push_latent(stand_logit)            
        
        # For post train
        if self.orth_mapper is not None:
            x = x @ self.orth_mapper.weight.T
            
        if 'both_feats' in self.extract_latent_place:
            if self.transfer_procrutes_mode:
                
                feat_list = [x]
            if self.latent_mapper_in is not None:
                x=x@self.latent_mapper_in
        
        if self.use_l2_norm_features:
            x = F.normalize(x, p=2, dim=-1)
                
        if self.transfer_mode:
            return x
        for layer in self.mlp_layers:
            if self.use_skip_connection:
                x = layer(x)+x
            else:
                x = alpha* layer(x) +x if self.transfer_direct_pred else layer(x)
            
        if self.use_l2_norm_features:
            x = F.normalize(x, p=2, dim=-1)
            
        if self.logit_norm_strategy == 'l2':
            x = F.normalize(x, p=2, dim=-1)
        if self.use_proj_ln:
            x = self.proj_ln(x)
        
        # POST MLP
        if self.extract_latent_place=='post_mlp':
            if self.transfer_procrutes_mode:
                return x
            
            if self.latent_mapper is not None:
                x = x@self.latent_mapper
               
        if 'both_feats' in self.extract_latent_place:
            if self.transfer_procrutes_mode:
                
                feat_list.append(x)
                return feat_list
            if self.latent_mapper_out is not None:
                x=x@self.latent_mapper_out
        
        # For post train
        if self.orth_mapper is not None:
            x = x @ self.orth_mapper.weight
        
        z = self.pull_latent(x)
        
        #POST LOGITS
        logits = self.transform_back_logits(logits,stand_logit, z, logit_mean, logit_std, alpha)

        if self.extract_latent_place=='post_logits':
            if self.transfer_procrutes_mode:
                return logits
            if self.latent_mapper is not None:
                logits = logits@self.latent_mapper
        
        logit_dicts['logits'] = logits
        return logit_dicts


class ProxyModel(nn.Module):
    def __init__(self, args, log=None):
        super(ProxyModel, self).__init__()
        
        self.val_preprocess = None
                    
        self.transfer_num_refine_layers = args.transfer_num_refine_layers
        self.network = nn.Sequential(*[LogitAdjustHead_ver3(args) for i in range(1)])
        
        self.max_interpolation = args.max_interpolation
        
        self.logit_norm_strategy = args.logit_norm_strategy
        self.class_padding_strategy = args.class_padding_strategy
        self.num_classes = args.num_classes
        self.use_interpolated_input = args.use_interpolated_input
        self.target_interpolate_ratio = args.target_interpolate_ratio
        
    def forward(self, input, logits, ft_logit=None):
        out_dict = {}
        
        if ft_logit is not None:
            zs_logprob = logits
            ft_logprob = ft_logit
            out_dict['interp_target'] = self.target_interpolate_ratio*ft_logprob+ (1-self.target_interpolate_ratio)*zs_logprob
        
        logit_dicts = self.network[0](logits)

        if isinstance(logit_dicts, dict):
            proxy_logit = logit_dicts['logits']
            if 'aux_logits' in logit_dicts:
                out_dict['aux_logits'] = logit_dicts['aux_logits']
            
        else:
            proxy_logit = logit_dicts

        out_dict["logits"] = proxy_logit

        return out_dict
