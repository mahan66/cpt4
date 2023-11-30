import os

import numpy as np
import timm.data
import torch
from robustbench.data import load_cifar10c, load_cifar100c
from torch import nn
from timm.models import create_model
import math
from torchvision import transforms, datasets
from tqdm import tqdm

DATASET_NAMES = {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'imagenet':'imagenet'}

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print(memory_available)
    print('chosen gpu:', np.argmax(memory_available))
    return np.argmax(memory_available)

def set_all_seeds(cur_seed: int) -> int:
    seed = cur_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        pbar = tqdm(range(n_batches), total=n_batches, leave=False)
        for counter in pbar:
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            x_curr = x_curr.to(device)
            y_curr = y_curr.to(device)

            #TODO ---- Basic Transformation-----
            x_curr = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])(x_curr)

            output = model(x_curr)
            # output = output['logits']
            num_correct = (output.max(1)[1] == y_curr).float().sum()
            acc += num_correct
            pbar.set_description(f"Step [{counter + 1}/{n_batches}], Num_correct/Num_cur_batch: {num_correct}/{x_curr.shape[0]}, Acc: {num_correct * 100 /x_curr.shape[0]:.2f}%")

    return acc.item() / x.shape[0]


def create_model_file_name(args):
    if not args.model_file_name:
        model_file_name = ''

        model_file_name += args.model
        model_file_name += '__' + args.dataset
        model_file_name += '__' + args.method_name
        model_file_name += '__' + 'pp'+str(int(args.prompt_pool))
        # model_file_name += 'b'+str(args.batch_size)+'__'
        model_file_name += '__' + 'tl_'+'-'.join(args.train_trainable_layers)
        if args.use_batch_norm:
            model_file_name += '__' + 'bl_'+'-'.join(args.batched_layers)
        model_file_name += '.npz'

        args.model_file_name = model_file_name

        if torch.cuda.is_available():
            # free_gpu_id = get_free_gpu()
            # device = torch.device('cuda:' + str(free_gpu_id))
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        args.device = device
    return args


def load_pretrained_model_state_dict(args):
    pretrained_model_file_name = os.path.join(args.model_dir_name, args.model_file_name)
    try:
        pretrianed_model_state_dict = torch.load(pretrained_model_file_name)
        return pretrianed_model_state_dict
    except Exception as e:
        print(str(e))
        return None


def configure_model(args, trainable_layers, test_time=False):
    # -------------- Original MODEL ---------------------
    if args.prompt_pool:
        print(f"Creating original model: {args.model}")
        original_model = create_model(
            args.model,
            pretrained=True,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
        original_model.to(args.device)
        original_model.requires_grad_(False)

    else:
        original_model = None

    # -------------- MODEL ---------------------
    print(f"Creating main model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )
    # Add batch norm layers in the specified layers
    if args.use_batch_norm:
        layer_names = args.batched_layers
        print(f'The model uses BatchNomrs in layers: {layer_names}.')
        model.add_batch_norm(layer_names=layer_names)

    # Load pretrained model based on model file name
    pretrained_model_state_dict = load_pretrained_model_state_dict(args)
    if pretrained_model_state_dict:
        model.load_state_dict(pretrained_model_state_dict)
        print(f'Pretrained model fetched successfully from this file: {args.model_file_name}')

    model.to(args.device)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Set trainable parameters
    if 'all' in trainable_layers:
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            # print(f'layer name: {n} and number of parameters: {torch.prod(torch.tensor(p.shape))}')
            for layer in trainable_layers:
                if layer in name:
                    param.requires_grad = True
                    break
            if test_time and 'batch_norm' in name:
                # force use of batch stats in train and eval modes
                param.track_running_stats = False
                param.running_mean = None
                param.running_var = None

    # learnable_parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    # head_parameters_count = sum(p.numel() for n, p in model.named_parameters() if 'head' in n)
    #
    # print(learnable_parameters_count, head_parameters_count)
    return model, original_model

def config_parser_with_config_name(config_name, config_parser):
    try:
        if config_name == 'cifar10_cpt4':
            from configs.cifar10_cpt4 import get_args_parser
        elif config_name == 'cifar100_cpt4':
            from configs.cifar100_cpt4 import get_args_parser

        get_args_parser(config_parser)
    except:
        print("Wrong Dataset Name")


def load_test_time_data(dataset: str, num_samples: int, severity: int, data_dir: str, shuffle: bool, corruption_types:list):
    x_test, y_test = None, None
    if dataset == DATASET_NAMES['cifar10']:
        x_test, y_test = load_cifar10c(num_samples, severity, data_dir, shuffle, corruption_types)
    elif dataset == DATASET_NAMES['cifar100']:
        x_test, y_test = load_cifar100c(num_samples, severity, data_dir, shuffle, corruption_types)
    elif dataset == DATASET_NAMES['imagenet']:
        # TODO To be implemented
        pass


    return x_test, y_test

def configure_datasets(dataset_name:str, data_path:str):
    train_dataset, test_dataset = None, None

    # transform = timm.data.create_transform(224)
    transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.CenterCrop(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if dataset_name == DATASET_NAMES['cifar10']:
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    elif dataset_name == DATASET_NAMES['cifar100']:
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    elif dataset_name == DATASET_NAMES['imagenet']:
        # TODO Required to be implemented
        pass



    return train_dataset, test_dataset
