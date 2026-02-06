import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, transmiters, args=None):
        super(EnsembleModel, self).__init__()
        self.transmiters = nn.ModuleList(transmiters)
        self.val_preprocess = None
        
        self.ensemble_strategy = args.ensemble_strategy
        self.num_classes = args.num_classes
        self.args = args
        
        if self.ensemble_strategy == 'learn_scale':
            self.learn_scale = nn.Parameter(torch.ones(len(self.transmiters)))
            
    def forward(self, input, logits,ft_logit=None, return_as_list=False):
        # only the output logit is considered
        outputs = [model(input,logits,ft_logit[i]) if ft_logit is not None \
            else model(input,logits) for i, model in enumerate(self.transmiters)]
        out_dict= {}
        # if self.training or return_as_list:
        #     for k in outputs[0].keys():
        #         out_dict[k] = [o[k] for o in outputs]
        # else:
        if self.ensemble_strategy == 'average':
            out_dict['logits'] = torch.stack([o['logits'] for o in outputs]).mean(0)
        elif self.ensemble_strategy == 'weighted':
            all_logits = torch.stack([o['logits'][:,:self.num_classes] for o in outputs])
            prob = all_logits.softmax(dim=-1)
            max_prob_per_model = prob.max(dim=-1,keepdim=True)[0]
            weight_per_model = max_prob_per_model.softmax(dim=0)
            out_dict['logits'] = (weight_per_model*all_logits).sum(dim=0)
        elif self.ensemble_strategy == 'learn_scale':
            all_logits = torch.stack([o['logits'][:,:self.num_classes] for o in outputs])
            learn_scale = self.learn_scale
            scale_learn = learn_scale.unsqueeze(-1).unsqueeze(-1).softmax(dim=0)
            out_dict['logits'] = (all_logits*scale_learn).sum(dim=0)
        elif self.ensemble_strategy == 'moe_logits':
            raise NotImplementedError(f"Ensemble strategy {self.ensemble_strategy} not implemented.")
        elif self.ensemble_strategy == 'moe_input_features':
            raise NotImplementedError(f"Ensemble strategy {self.ensemble_strategy} not implemented.")
        elif self.ensemble_strategy == 'moe_output_features':
            raise NotImplementedError(f"Ensemble strategy {self.ensemble_strategy} not implemented.")
        return out_dict