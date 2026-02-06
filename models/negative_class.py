import os
import random
import torch

from datasets.registry import get_dataset
from nltk.corpus import wordnet as wn
import numpy as np

from models.heads import get_word_embeddings, get_class_embeddings
from datasets.templates import get_templates, article

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
from datasets.common import get_dataloader, maybe_dictionarize
from models.logit_stand import logit_stand

def negative_class_selection(train_dataset, args, logger, use_saved=False):
    dat = get_dataset(
            train_dataset,
            None,
            location=args.data_location,
            batch_size=args.batch_size,
            args = args
        )
    num_classes = len(dat.classnames)
    args.num_classes = num_classes
    num_neg_classes = args.proj_dim - num_classes if args.tot_logit_dim ==-1 else args.tot_logit_dim - num_classes
    args.num_pad = num_neg_classes
    
    # If use_saved is True and selected_words already exists, skip selection
    if use_saved and hasattr(args, 'selected_words') and args.selected_words is not None:
        logger.info(f"Using pre-loaded auxiliary classes: {len(args.selected_words)} words")
        return args
    
    args.selected_words = None
        
    if num_neg_classes<=0:
        logger.info(f"No need to select negative classes")
        args.selected_words = None
        return args
    
    if args.class_padding_strategy in ['wordnet','openimage']:
        random.seed(args.seed)
        
        if args.class_padding_strategy == 'wordnet':
            # num_classes = len(dat.classnames)
            object_entity = wn.synset('object.n.01')
            all_hyponyms = list(object_entity.closure(lambda s: s.hyponyms()))
            object_words = set()
            for synset in tqdm(all_hyponyms):
                for lemma in synset.lemmas():
                    object_words.add(lemma.name())
            object_words = list(object_words)
        elif args.class_padding_strategy == 'openimage':
            df = pd.read_csv("data/classnames_dictionay.csv")
            display_names = df['ClassName']
            object_words =  display_names.tolist()
        else:
            raise NotImplementedError
        logger.info(f"Total number of objects in {args.class_padding_strategy}: {len(object_words)}")
        object_words = sorted([word for word in object_words if word not in dat.classnames])
        if args.class_sampling_strategy=='random':
            selected_words = random.sample(object_words, num_neg_classes)
        elif args.class_sampling_strategy=='all':
            selected_words = object_words
        elif args.class_sampling_strategy=='top_std':
            class_embeddings = get_class_embeddings(args.text_encoder, args.eval_datasets, dat, device=args.device, args=args)
            word_embeddings = get_word_embeddings('clip', args.eval_datasets, object_words, device=args.device, args=args)
            dataset = get_dataset(
                train_dataset,
                None,
                location=args.data_location,
                batch_size=args.batch_size,
                is_transfer=True,
                use_train_for_proxy=args.use_train_for_proxy,
                data_ratio=args.data_ratio,
                args=args
            )
            all_logits = run_models(dataset,word_embeddings,args)
            # all_logits = run_models(dataset,torch.cat([class_embeddings,word_embeddings]),args)
            # all_logits = logit_stand(all_logits, return_others=False)[:,num_classes:]
            class_wise_std = all_logits.std(dim=0)
            topk_class_indices = torch.topk(class_wise_std, k=num_neg_classes, largest=True).indices
            selected_words = [object_words[i] for i in topk_class_indices]
        elif args.class_sampling_strategy=='std_fps':
            class_embeddings = get_class_embeddings(args.text_encoder, args.eval_datasets, dat, device=args.device, args=args)
            word_embeddings = get_word_embeddings('clip', args.eval_datasets, object_words, device=args.device, args=args)
            dataset = get_dataset(
                train_dataset,
                None,
                location=args.data_location,
                batch_size=args.batch_size,
                is_transfer=True,
                use_train_for_proxy=args.use_train_for_proxy,
                data_ratio=args.data_ratio,
                args=args
            )
            all_logits = run_models(dataset,torch.cat([class_embeddings,word_embeddings]),args)
            # all_logits = logit_stand(all_logits, return_others=False)
            
            class_wise_std = all_logits.std(dim=0)
            topk_class_indices = torch.topk(class_wise_std[num_classes:], k=all_logits.shape[1]//4, largest=True).indices
            task_logit_embeddings = all_logits.T[:num_classes]
            aux_logit_embeddings = all_logits.T[num_classes:]
            topk_std_embeddings = aux_logit_embeddings[topk_class_indices] # K x num_samples
            
            selected_indices = fps_with_fixed_optimized(torch.cat([task_logit_embeddings,topk_std_embeddings]), num_classes, num_neg_classes+num_classes, dist_metric='euclidean')[num_classes:]
            
            selected_words = [object_words[topk_class_indices[i-num_classes]] for i in selected_indices]
            
            
        elif args.class_sampling_strategy=='fps':
            class_embeddings = get_class_embeddings(args.text_encoder, args.eval_datasets, dat, device=args.device, args=args)
            # we_dir = os.path.join(args.word_embedding_dir, 'wordnet',args.text_encoder)
            # if os.path.exists(we_dir):
            #     word_embeddings = np.load(os.path.join(we_dir,'embeddings.npy'))
            #     word_embeddings = torch.tensor(word_embeddings).to(args.device)
            # else:
            word_embeddings = get_word_embeddings(args.text_encoder, args.eval_datasets, object_words, device=args.device, args=args)
            # os.makedirs(we_dir, exist_ok=True)
            # np.save(os.path.join(we_dir,'embeddings.npy'), word_embeddings.cpu().numpy())
            filtered_word_indices = [i for i,word in enumerate(object_words)]
            filtered_words = [object_words[i] for i in filtered_word_indices]
            word_embeddings = word_embeddings[filtered_word_indices]
            
            selected_indices = fps_with_fixed_optimized(torch.cat([class_embeddings, word_embeddings]), num_classes, num_neg_classes+num_classes)
            additional_indices = selected_indices[num_classes:]
            selected_words = [filtered_words[i-num_classes] for i in additional_indices]
        elif args.class_sampling_strategy=='mix_fps':
            
            class_embeddings = get_class_embeddings(args.text_encoder, args.eval_datasets, dat, device=args.device, args=args)
            word_embeddings = get_word_embeddings(args.text_encoder, args.eval_datasets, object_words, device=args.device, args=args)
            
            filtered_word_indices = [i for i,word in enumerate(object_words)]
            filtered_words = [object_words[i] for i in filtered_word_indices]
            word_embeddings = word_embeddings[filtered_word_indices]
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            word_embeddings = word_embeddings / word_embeddings.norm(dim=-1, keepdim=True)
            
            dist_matrix = class_embeddings@word_embeddings.T
            
            sim = dist_matrix.max(dim=0).values
            num_words = len(object_words)
            top_25_percent_indices = torch.topk(sim, k=num_words // 4, largest=True).indices
            bottom_25_percent_indices = torch.topk(sim, k=num_words // 4, largest=False).indices

            top_25_percent_embeddings = word_embeddings[top_25_percent_indices]
            bottom_25_percent_embeddings = word_embeddings[bottom_25_percent_indices]

            pos_sample = (num_neg_classes) // 2
            neg_sample = num_neg_classes - pos_sample

            top_selected_indices = fps_with_fixed_optimized(
                torch.cat([class_embeddings, top_25_percent_embeddings]), num_classes, num_classes + pos_sample
            )[num_classes:]

            bottom_selected_indices = fps_with_fixed_optimized(
                torch.cat([class_embeddings, bottom_25_percent_embeddings]), num_classes, num_classes + neg_sample
            )[num_classes:]

            selected_words = [filtered_words[top_25_percent_indices[i-num_classes].item()] for i in top_selected_indices] + \
                                [filtered_words[bottom_25_percent_indices[i-num_classes].item()] for i in bottom_selected_indices]
        
        elif args.class_sampling_strategy=='easy_fps':
            class_embeddings = get_class_embeddings(args.text_encoder, args.eval_datasets, dat, device=args.device, args=args)
            word_embeddings = get_word_embeddings(args.text_encoder, args.eval_datasets, object_words, device=args.device, args=args)
            
            filtered_word_indices = [i for i,word in enumerate(object_words)]
            filtered_words = [object_words[i] for i in filtered_word_indices]
            word_embeddings = word_embeddings[filtered_word_indices]
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            word_embeddings = word_embeddings / word_embeddings.norm(dim=-1, keepdim=True)
            
            dist_matrix = class_embeddings@word_embeddings.T
            
            sim = dist_matrix.max(dim=0).values
            num_words = len(object_words)
            top_25_percent_indices = torch.topk(sim, k=num_words // 4, largest=True).indices
            # bottom_25_percent_indices = torch.topk(sim, k=num_words // 4, largest=False).indices

            top_25_percent_embeddings = word_embeddings[top_25_percent_indices]
            # bottom_25_percent_embeddings = word_embeddings[bottom_25_percent_indices]

            top_selected_indices = fps_with_fixed_optimized(
                torch.cat([class_embeddings, top_25_percent_embeddings]), num_classes, num_classes + num_neg_classes
            )[num_classes:]

            # bottom_selected_indices = fps_with_fixed_optimized(
            #     torch.cat([class_embeddings, bottom_25_percent_embeddings]), num_classes, num_classes + neg_sample
            # )[num_classes:]

            selected_words = [filtered_words[top_25_percent_indices[i-num_classes].item()] for i in top_selected_indices]
        
        elif args.class_sampling_strategy=='hard_neg':
            class_embeddings = get_class_embeddings(args.text_encoder, args.eval_datasets, dat, device=args.device, args=args)
            word_embeddings = get_word_embeddings(args.text_encoder, args.eval_datasets, object_words, device=args.device, args=args)
            
            filtered_word_indices = [i for i,word in enumerate(object_words)]
            filtered_words = [object_words[i] for i in filtered_word_indices]
            word_embeddings = word_embeddings[filtered_word_indices]
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            word_embeddings = word_embeddings / word_embeddings.norm(dim=-1, keepdim=True)
            
            dist_matrix = class_embeddings@word_embeddings.T
            
            sim = dist_matrix.max(dim=0).values
            neg_class_indices = torch.topk(sim, k=num_neg_classes, largest=True).indices
            selected_words = [filtered_words[i] for i in neg_class_indices]
            
        else:
            raise NotImplementedError
        
        args.selected_words = selected_words
        logger.info(f"Selected words in {args.class_padding_strategy}: {selected_words}")
        
    elif args.class_padding_strategy == 'multi_prompt':
        # template = get_templates(train_dataset)
        template = get_templates(args.class_padding_strategy)
        # dataset = get_dataset(
        #     train_dataset,
        #     None,
        #     location=args.data_location,
        # )

        # num_classes = len(dataset.classnames)
        
        args.num_pad = num_neg_classes
        assert args.num_pad >= 0
        all_texts = []
        for class_name in dat.classnames:
            texts = []
            for t in template:
                texts.append(t(class_name))
            all_texts.extend(texts)
        if len(all_texts) < args.num_pad:
            logger.info("=======================================================================================")
            logger.info(f"Warning: Number of texts {len(all_texts)} is less than number of padding {args.num_pad}")
            logger.info("=======================================================================================")
        
        if args.class_sampling_strategy=='random':
            random.seed(args.seed)
            args.selected_words = random.sample(all_texts, min(args.num_pad, len(all_texts)))
        elif args.class_sampling_strategy=='fps':

            # args.selected_words = all_texts
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    elif args.class_padding_strategy =='random':
        # dat = get_dataset(
        #     train_dataset,
        #     None,
        #     location=args.data_location,
        #     batch_size=args.batch_size,
        # )
        args.num_pad = num_neg_classes
    elif args.class_padding_strategy == 'description':
        args.class_padding_strategy
        description_path = 'third_party/adaptclipzs/gpt_descriptions'
        description_path = os.path.join(description_path, description_dict[train_dataset])
        
        # args.num_pad = num_neg_classes
        description_files = os.listdir(description_path)
        if train_dataset=='DTD':
            matching_files = []
            for classname in dat.classnames:
                for file in description_files:
                    file_name = file.replace('_',' ')
                    if classname in file_name or classname in file:
                        matching_files.append(file)
                        break
        else:
            matching_files = description_files
        
        if len(matching_files) < num_classes:
            logger.warning(f"Not enough matching files found. Required: {num_classes}, Found: {len(matching_files)}")
        
        # selected_files = random.sample(matching_files, min(num_neg_classes, len(matching_files)))
        # args.selected_words = selected_files
        all_descriptions = []
        num_description_per_class = []
        for file in matching_files:
             
            with open(os.path.join(description_path, file), 'r') as f:
                all_descriptions.extend(f.readlines())
            with open(os.path.join(description_path, file), 'r') as f:
                num_description_per_class.append(len(f.readlines()))
        
        
        if len(all_descriptions) < num_neg_classes:
            logger.warning(f"Not enough descriptions found. Required: {num_neg_classes}, Found: {len(all_descriptions)}")
        
        all_descriptions = [desc.strip() for desc in all_descriptions]
        all_descriptions = [desc.split(' ',1)[1][:-1] for desc in all_descriptions]
        
        if args.class_sampling_strategy=='random':
            random.seed(args.seed)
            selected_descriptions = random.sample(all_descriptions, min(num_neg_classes, len(all_descriptions)))
            args.tot_logit_dim = num_classes + len(selected_descriptions)
            args.selected_words = selected_descriptions
        elif args.class_sampling_strategy=='fps':
            # class_embeddings = get_class_embeddings(args.text_encoder, args.eval_datasets, dat, device=args.device, args=args)
            word_embeddings = get_word_embeddings(args.text_encoder, args.eval_datasets, all_descriptions, device=args.device, args=args)
            
            # select first description
            init_indices = 0
            init_embedding_list= []
            init_indice_list = []
            for n_desc in num_description_per_class:
                init_embedding_list.append(word_embeddings[init_indices])
                init_indice_list.append(init_indices)
                init_indices += n_desc
                
            # init_embeddings = torch.stack(init_embedding_list)
            
            assert num_neg_classes >= num_classes, f"Number of negative classes should be greater than number of classes. Got {num_neg_classes} and {num_classes}"
            selected_indices = fps_with_fixed_optimized(word_embeddings, num_classes, num_neg_classes, init_indice_list=init_indice_list)
            # raise NotImplementedError
            args.selected_words = [all_descriptions[i] for i in selected_indices]
            args.tot_logit_dim = num_classes + len(selected_indices)
        elif args.class_sampling_strategy=='uniform':
            
            raise NotImplementedError
        elif args.class_sampling_strategy=='all':
            args.selected_words = all_descriptions
            args.tot_logit_dim = num_classes + len(all_descriptions)
        
        logger.info(f'number of descriptions: {len(args.selected_words)}')
        
        
        # selected_descriptions = random.sample(all_descriptions, min(num_neg_classes, len(all_descriptions)))
        # args.selected_words = selected_descriptions
        
        
    return args

def run_models(dataset,word_embeddings,args):
    
    data_loader = get_dataloader(
            dataset,is_train=True, args=args, image_encoder=None, is_transfer=True)
    all_logits = []
    with torch.no_grad():
        for i,data in enumerate(tqdm(data_loader)):
            data = maybe_dictionarize(data)
            if args.use_precompute_features:
                x = None
                source_feats = data[f'{args.source_model}_features'].to(args.device)
                source_feats /= source_feats.norm(dim=-1, keepdim=True)
                source_feats = source_feats.to(dtype=word_embeddings.dtype)
                source_logit = source_feats@word_embeddings.T
                
            else:
                raise NotImplementedError
            all_logits.append(source_logit)
        all_logits = torch.cat(all_logits)
    return all_logits
    

def fps_with_fixed_optimized(embeddings, num_class, K,init_indice_list=None, dist_metric='cosine'):
    M, D = embeddings.shape
    device = embeddings.device
    
    # selected_indices = torch.tensor(fixed_indices, device=device)
    if init_indice_list is not None:
        fixed_indices = init_indice_list
        selected_indices = torch.tensor(fixed_indices, device=device)
    else:
        selected_indices = torch.arange(num_class, device=device)
        fixed_indices = selected_indices.cpu().tolist()
    N = num_class
    
    similarities = torch.full((M,), -1.0, device=device)

    if dist_metric == 'cosine':
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    else:
        # embeddings = logit_stand(embeddings, return_others=False)
        embeddings = embeddings
    
    selected_mask = torch.zeros(M, dtype=torch.bool, device=device)
    selected_mask[fixed_indices] = True

    fixed_embeddings = embeddings[fixed_indices]  # (N, D)
    
    if dist_metric == 'cosine':
        sims = (embeddings @ fixed_embeddings.T)  # (M, N)
    else:
        sims = -torch.cdist(embeddings, fixed_embeddings, p=2)  # (M, N)
    max_sims, _ = sims.max(dim=1) 
    similarities = torch.maximum(similarities, max_sims)
    
    for _ in range(K - N):
        similarities_selected = similarities.clone()
        similarities_selected[selected_mask] = float('inf')
        next_index = torch.argmin(similarities_selected)
        selected_indices = torch.cat([selected_indices, next_index.unsqueeze(0)])
        selected_mask[next_index] = True

        new_embedding = embeddings[next_index]  # (D,)
        if dist_metric == 'cosine':
            sims = embeddings @ new_embedding  # (M,)
        else:
            sims = -torch.cdist(embeddings, new_embedding.unsqueeze(0), p=2).squeeze()
        similarities = torch.maximum(similarities, sims)
    
    return selected_indices.tolist()

description_dict={
    'DTD':'gpt4_0613_api_DTD',
    'CUB':'gpt4_0613_api_CUB',
    'SUN397':'gpt4_0613_api_Sun397',
    'Cars':'gpt4_0613_api_StanfordCars',
    'EuroSAT':'gpt4_0613_api_EuroSAT',
    'Flowers':'gpt4_0613_api_Flowers102',
    'Food101':'gpt4_0613_api_Food101',
    'Pets':'gpt4_0613_api_OxfordIIITPets',
    'UCF':'gpt4_0613_api_UCF101',
    'Aircraft':'gpt4_0613_api_FGVCAircraft',
    'Caltech':'gpt4_0613_api_CalTech101',
    'ImageNet':'gpt4_0613_api_ImageNet',
    
}