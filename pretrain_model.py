import argparse
import math
import os
import time

import numpy as np
import timm.data
import torch
import torch.optim as optim
from timm.models import create_model
from timm.utils import accuracy
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from torchvision import transforms

from utils import configure_model
from utils import set_all_seeds

import models
import utils


def pretrain(args):
    pretrained_models_path = args.model_dir_name
    if not os.path.exists(pretrained_models_path):
        os.makedirs(pretrained_models_path)
        print(f"Directory '{pretrained_models_path}' created.")
    else:
        print(f"Directory '{pretrained_models_path}' already exists.")

    set_all_seeds(args.seed)
    device = torch.device(args.device)

    model, original_model = configure_model(args, trainable_layers=args.train_trainable_layers)
    train_loader, test_loader, validation_loader = create_datasets(args)
    best_model = train_model(model, original_model, train_loader, validation_loader, device, args)
    evaluate_model(best_model, original_model, test_loader, device, args)


def create_datasets(args):

    train_dataset, test_dataset = utils.configure_datasets(args.dataset, args.data_path)

    ratio = args.pretrain_ratio
    validation_ratio = 0.1

    num_train = len(train_dataset)
    num_test = len(test_dataset)
    train_indices = list(range(num_train))
    test_indices = list(range(num_test))
    train_split = int(ratio * num_train)
    test_split = int(ratio * num_test)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    train_idx_cur = train_indices[:train_split]
    test_idx = test_indices[:test_split]

    num_train = len(train_idx_cur)
    num_validation = int(validation_ratio * num_train)
    validation_idx = train_idx_cur[:num_validation]
    train_idx = train_idx_cur[num_validation:]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    validation_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=validation_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)

    return train_dataloader, validation_dataloader, test_dataloader


def train_model(model, original_model, train_loader, validation_loader, device, args):
    # Define hyperparameters
    num_epochs = 20 if args.epochs is None else args.epochs
    learning_rate = 1e-3 if not args.lr else args.lr
    weight_decay = 1e-4
    step_size = 10 if not args.lr_step_size else args.lr_step_size
    gamma = 0.5 if not args.lr_gamma else args.lr_gamma
    best_valid_loss = math.inf
    best_model = model

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.train()
    total_step = len(train_loader)
    for e in range(num_epochs):
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        pbar = tqdm(enumerate(train_loader), total=total_step, leave=False)
        for batch_idx, (input, target) in pbar:
            input = input.to(device)
            target = target.to(device)

            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

            output = model(input, cls_features=cls_features, train=True)
            logits = output['logits']

            loss = criterion(logits, target)
            cur_loss = loss
            if args.pull_constraint and 'reduce_sim' in output:
                loss = loss + args.pull_constraint_coeff * pow(output['reduce_sim'], 2)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.synchronize()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            elapsed = time.time() - start_time

            # print(f" Class_loss: {cur_loss:.4f}, Reduce_sim: {output['reduce_sim']:.4f}\n")

            pbar.set_description(f"Epoch [{e+1}/{num_epochs}], Step [{batch_idx+1}/{total_step}], "
                                 f"Duration: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}, "
                                 f"Loss: {loss.item():.4f}, Acc1: {acc1:.2f}, Acc5: {acc5:.2f}")
                                 # f"Class_loss: {cur_loss:.4f}, Reduce_sim: {output['reduce_sim']:.4f}")
            # print(f'loss: {loss}, acc1: {acc1}, acc5: {acc5}')
        scheduler.step()

        # -------------------------- Validation Section -----------------
        train_accuracy = 100 * correct / total
        train_loss = train_loss / total_step
        valid_loss, valid_acc = evaluate_model(model, original_model, validation_loader, device, args, eval_type='Validation')
        print(f"Epoch [{e + 1}/{num_epochs}], "
              f"Duration: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}, "
              f"Validation Acc: {valid_acc:.2f}%, "
              f"Validation Loss: {valid_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Train Loss: {train_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_model = model
            best_valid_loss = valid_loss
            model_output_name = args.model_output_name if args.model_output_name else args.model_file_name
            output_model_path = os.path.join(args.model_dir_name, model_output_name)
            torch.save(model.state_dict(), output_model_path)

    return best_model


def evaluate_model(model, original_model, cur_loader, device, args, eval_type='Test'):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    criterion = torch.nn.CrossEntropyLoss().to(device)
    total_step = len(cur_loader)
    pbar = tqdm(enumerate(cur_loader), total=total_step, leave=False)
    for i, (input, target) in pbar:
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

            output = model(input, cls_features=cls_features, train=False)
            logits = output['logits']
            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            total_loss += loss.item()

            pbar.set_description(f"Step [{i + 1}/{total_step}], Acc1: {acc1:.3f}, Acc5: {acc5:.3f}")

    acc = correct*100/total
    avg_loss = total_loss/total_step
    if eval_type == 'Test':
        print(f"{eval_type} Accuracy: {acc:.2f}%, {eval_type} Loss: {avg_loss:.4f}")
    return avg_loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining Configs')
    config_name = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest=config_name)

    config_parser = subparser.add_parser(config_name)

    utils.config_parser_with_config_name(config_name, config_parser)

    args = parser.parse_args()

    args = utils.create_model_file_name(args)

    pretrain(args)
