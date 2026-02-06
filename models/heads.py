"""
Classification head building utilities for TransMITER.
"""

import os
import torch
from tqdm import tqdm
import open_clip

from datasets.templates import get_templates, wordnet_templates
from datasets.registry import get_dataset
from models.modeling import ClassificationHead, ImageEncoder
from transformers import AutoTokenizer, AutoModel
from datasets.few_shot import data_name_dict


description_dict = {
    'DTD': 'gpt4_0613_api_DTD',
    'CUB': 'gpt4_0613_api_CUB',
    'SUN397': 'gpt4_0613_api_Sun397',
    'Cars': 'gpt4_0613_api_StanfordCars',
    'EuroSAT': 'gpt4_0613_api_EuroSAT',
    'Flowers': 'gpt4_0613_api_Flowers102',
    'Food101': 'gpt4_0613_api_Food101',
    'Pets': 'gpt4_0613_api_OxfordIIITPets',
    'UCF': 'gpt4_0613_api_UCF101',
    'Aircraft': 'gpt4_0613_api_FGVCAircraft',
    'Caltech': 'gpt4_0613_api_CalTech101',
    'ImageNet': 'gpt4_0613_api_ImageNet',
}


def load_description(dataset_name, args):
    description_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'gpt_descriptions')
    description_folder = os.path.join(description_path, description_dict[dataset_name])
    
    descriptions = {}
    for filename in os.listdir(description_folder):
        if filename.endswith('.txt'):
            classname = filename[:-4]
            if 'SLASH' in classname:
                classname = classname.replace('SLASH', '/')
            with open(os.path.join(description_folder, filename), 'r') as file:
                description = file.read().splitlines()
                for i, (desc) in enumerate(description):
                    description[i] = desc[0].lower() + desc[1:]
            descriptions[classname] = []
            for desc in description:
                descriptions[classname].append(lambda c, desc=desc: f'a photo of {c} with {desc}')
    return descriptions


def build_classification_head(model, dataset_name, data_location, device, args=None):
    if args.setting is not None:
        template = get_templates('few_shot')[data_name_dict[dataset_name]]
        
        if args.init_template is not None:
            if args.init_template == 'openai':
                template = get_templates(args.init_template)[data_name_dict[dataset_name]]
            elif args.init_template == 'description':
                all_templates = load_description(dataset_name, args)
            elif args.init_template == 'llmbo':
                template = get_templates(args.init_template)[data_name_dict[dataset_name]]
            else:
                template = get_templates(args.init_template)
    else:
        template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location,
        args=args
    )
    model.eval()
    model.to(device)

    if '__pretrained__' in args.model:
        tokenizer = open_clip.get_tokenizer(args.model.split('__pretrained__')[0])
    else:
        tokenizer = open_clip.tokenize
    
    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            if args.init_template == 'description':
                template = all_templates[classname]
            for t in template:
                texts.append(t(classname))
            
            texts = tokenizer(texts).to(device)
            embeddings = model.encode_text(texts)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= logit_scale.exp()
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
        
        if args is not None and args.selected_words is not None:
            if args.class_padding_strategy in ['wordnet', 'description', 'openimage']:
                if args.aux_templates is not None:
                    aux_template = get_templates('main_aux_templates')[args.aux_templates]
                else:
                    aux_template = template
                wordnet_embeddings = []
                for word in tqdm(args.selected_words):
                    wordnet_texts = []
                    for t in aux_template:
                        wordnet_texts.append(t(word))
                    
                    wordnet_texts = tokenizer(wordnet_texts).to(device)
                    wordnet_embedding = model.encode_text(wordnet_texts)
                    wordnet_embedding /= wordnet_embedding.norm(dim=-1, keepdim=True)
                    wordnet_embedding = wordnet_embedding.mean(dim=0, keepdim=True)
                    wordnet_embedding /= wordnet_embedding.norm()
                    wordnet_embeddings.append(wordnet_embedding)
                wordnet_embeddings = torch.stack(wordnet_embeddings, dim=0).to(device)
                wordnet_embeddings = torch.transpose(wordnet_embeddings, 0, 2)
                wordnet_embeddings *= logit_scale.exp()
                wordnet_embeddings = wordnet_embeddings.squeeze().float()
                wordnet_embeddings = torch.transpose(wordnet_embeddings, 0, 1)
                zeroshot_weights = torch.cat([zeroshot_weights, wordnet_embeddings], dim=0)
            elif args.class_padding_strategy == 'multi_prompt':
                prompt_embeddings = open_clip.tokenize(args.selected_words).to(device)
                prompt_embeddings = model.encode_text(prompt_embeddings)
                prompt_embeddings /= prompt_embeddings.norm(dim=-1, keepdim=True)
                prompt_embeddings *= logit_scale.exp()
                zeroshot_weights = torch.cat([zeroshot_weights, prompt_embeddings], dim=0)
            elif args.class_padding_strategy == 'random':
                random_generator = torch.Generator().manual_seed(args.seed)
                random_embeddings = torch.randn(args.num_pad, zeroshot_weights.shape[1], generator=random_generator).to(device)
                random_embeddings /= random_embeddings.norm(dim=-1, keepdim=True)
                random_embeddings *= logit_scale.exp()
                zeroshot_weights = torch.cat([zeroshot_weights, random_embeddings], dim=0)
                
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    return classification_head


def get_classification_head(args, dataset, save_dir=None):
    if save_dir is None:
        filename = os.path.join(args.save, f'head_{dataset}.pt')
    else:
        filename = os.path.join(save_dir, f'head_{dataset}.pt')
    
    if args.class_padding_strategy == 'none' or (args.setting is None):
        print(f'Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch.')
    else:
        print(f'Building classification head with padding strategy {args.class_padding_strategy}')
    
    model = ImageEncoder(args, keep_lang=True).model
    classification_head = build_classification_head(model, dataset, args.data_location, args.device, args)
    
    if args.class_padding_strategy == 'none' and not args.feature_extract and (args.setting is None):
        if save_dir is None:
            os.makedirs(args.save, exist_ok=True)
        else:
            os.makedirs(save_dir, exist_ok=True)
        classification_head.save(filename)
    return classification_head


def get_class_embeddings(model_n, dataset_name, dataset, device, args=None):
    if args.use_dtemp_for_neg:
        template = get_templates('aux_templates')[data_name_dict[dataset_name]]
    else:
        template = wordnet_templates[args.padding_template]
    
    if model_n == 'clip':
        if '__pretrained__' in args.source_model:
            name, pretrained = args.source_model.split('__pretrained__')
            tokenizer = open_clip.get_tokenizer(name)
        else:
            name = args.source_model
            pretrained = 'openai'
            tokenizer = open_clip.tokenize
        model = open_clip.create_model_and_transforms(name, pretrained=pretrained)[0]
    else:
        if model_n == 'bert':
            model_name = 'bert-large-uncased'
        elif model_n == 'roberta':
            model_name = 'roberta-large'
        else:
            raise NotImplementedError(f"Model {model_n} not implemented")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    
    model.eval()
    model.to(device=device)
    with torch.no_grad():
        dataset_classnames = [template(c) for c in dataset.classnames]
        if model_n == 'clip':
            inputs = tokenizer(dataset_classnames).to(device)
            outputs = model.encode_text(inputs)
            outputs /= outputs.norm(dim=-1, keepdim=True)
        else:
            inputs = tokenizer(dataset_classnames, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs).pooler_output
        zeroshot_weights = outputs.detach()
    del model
    
    return zeroshot_weights


def get_word_embeddings(model_n, dataset_name, object_words, device, args=None):
    if args.use_dtemp_for_neg:
        template = get_templates('aux_templates')[data_name_dict[dataset_name]]
    else:
        template = wordnet_templates[args.padding_template]
    
    if model_n == 'clip':
        model = open_clip.create_model_and_transforms(args.source_model, pretrained='openai')[0]
    else:
        if model_n == 'bert':
            model_name = 'bert-large-uncased'
        elif model_n == 'roberta':
            model_name = 'roberta-large'
        else:
            raise NotImplementedError(f"Model {model_n} not implemented")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    
    model.eval()
    model.to(device=device)
    with torch.no_grad():
        zeroshot_weights = []
        batch_size = 1000
        for i in tqdm(range(0, len(object_words), batch_size)):
            batch_words = object_words[i:i + batch_size]
            batch_words = [template(c) for c in batch_words]
            if model_n == 'clip':
                inputs = open_clip.tokenize(batch_words).to(device)
                outputs = model.encode_text(inputs)
                outputs /= outputs.norm(dim=-1, keepdim=True)
                text_features = outputs.detach()
            else:
                inputs = tokenizer(batch_words, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model(**inputs).pooler_output
                text_features = outputs.detach()
            zeroshot_weights.append(text_features)
    del model
    
    return torch.cat(zeroshot_weights, dim=0)
