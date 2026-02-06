import torch
import numpy as np

def add_noise(logits,args):
    assert args.noise_alpha != 0
    # generator = torch.Generator(device=logits.device).manual_seed(args.seed)
    if np.random.uniform() >0.5:
        if args.class_wise_noise:
            cw_std = args.source_std['class_wise_logit_std'].unsqueeze(0).repeat(logits.shape[0],1)
            noise = torch.normal(mean=0., std=args.noise_alpha*cw_std)
            
            
        else:
            logit_std = logits.std(dim=-1, keepdim=True).repeat(1, logits.shape[-1])
            noise = torch.normal(mean=0., std=args.noise_alpha*logit_std)
        return noise + logits
    else:
        return logits