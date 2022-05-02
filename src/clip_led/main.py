import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import json 
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

from torchvision import transforms
from tqdm import tqdm
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import sys 
sys.path.append('../..')
import src.utils as utils
import src.clip as clip 
import yaml
import math 
from tqdm import tqdm  
from src.clip_led.dataset import LEDDataset
from src.clip_led.model import LEDModel 
from src.clip_led.engine import train_model, eval_model
import src.fusion as fusion
from src.blocks import Up, ConvBlock, IdentityBlock


# Change this to YAML
config = {
    # Data Paths
    'train_path' : '../../data/way_splits/train_debug_data.json',
    'valid_seen_path' : '../../data/way_splits/valSeen_debug_data.json',
    'valid_unseen_path': '../../data/way_splits/valUnseen_debug_data.json',
    'mesh2meters': '../../data/floorplans/pix2meshDistance.json',
    'image_dir': '../../data/floorplans/',
    'geodistance_file': '../../data/geodistance_nodes.json',
    'save_path': '../../logs/checkpoints',

    'device': 'cuda:0',

    # Hyper Parameters
    'max_floors': 5,

    # Image Parameters
    'image_size': [3, 448, 448],
    # 'image_size': [3, 700, 1200],
    'original_image_size': [3, 700, 1200],
    'cropped_image_size': [3, 700, 800],
    'scaled_image_size': [3, 448, 448],


    'crop_translate_x': 200,
    'crop_translate_y': 0,
    'resize_scale_x': 448/800,
    'resize_scale_y': 448/700,
    'conversion_scale': 448/800,


    'lang_fusion_type': 'mult',
    'num_post_clip_channels': 2048, 
    'bilinear': True,
    'batch_norm': True, 
    'num_output_channels': 1,

    'lr': 0.001,
}

# Training Loop 


def training_loop(train_loader, valid_seen_loader, valid_unseen_loader, epochs, model, loss_fn, optimizer, scaler, scheduler, config):

    # Metrics 
    metrics = {
        'train_loss': 0,
        'valid_seen_loss': 0,
        'valid_unseen_loss': 0,
        'train_acc_5m': 0, 
        'train_acc_3m': 0, 
        'train_acc_0m': 0, 
        'valid_seen_acc_5m': 0, 
        'valid_seen_acc_3m': 0, 
        'valid_seen_acc_0m': 0, 
        'valid_unseen_acc_5m': 0,
        'valid_unsseen_acc_3m': 0,
        'valid_unsseen_acc_0m': 0,
    }
    best_loss = float('inf')
    # Training 
    for e in range(epochs): 

        model.train()
        train_metrics = train_model(model, train_loader, loss_fn, optimizer, scaler, config)
        
        print(f'Train Loss: {train_metrics["loss"]}')
        print(f'Train Acc0m: {train_metrics["acc0m"]}')
        print(f'Train Acc3m: {train_metrics["acc3m"]}')
        print(f'Train Acc5m: {train_metrics["acc5m"]}')
        
        utils.assign_metrics(metrics, train_metrics, 'train')

        model.eval()

        valid_seen_metrics = eval_model(model, valid_seen_loader, loss_fn, config, 'valid_seen')

        print(f'Valid Seen Loss: {valid_seen_metrics["loss"]}')
        print(f'Valid Seen Acc0m: {valid_seen_metrics["acc0m"]}')
        print(f'Valid Seen Acc3m: {valid_seen_metrics["acc3m"]}')
        print(f'Valid Seen Acc5m: {valid_seen_metrics["acc5m"]}')

        utils.assign_metrics(metrics, valid_seen_metrics, 'valid_seen')

        valid_unseen_metrics = eval_model(model, valid_seen_loader, loss_fn, config, 'valid_unseen')

        print(f'Valid Unseen Loss: {valid_seen_metrics["loss"]}')
        print(f'Valid Unseen Acc0m: {valid_seen_metrics["acc0m"]}')
        print(f'Valid Unseen Acc3m: {valid_seen_metrics["acc3m"]}')
        print(f'Valid Unseen Acc5m: {valid_seen_metrics["acc5m"]}')

        utils.assign_metrics(metrics, valid_unseen_metrics, 'valid_unseen')

        print(metrics)

        if metrics['valid_unseen_loss'] < best_loss:
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_path = config['save_path'] + f'_epoch_{e}_loss_{metrics["valid_unseen_loss"]}.pth'
            torch.save(save_dict, save_path)
            best_loss = metrics['valid_unseen_loss']
        
        scheduler.step(metrics['valid_unseen_loss'])

        


def main():
    train_dataset = LEDDataset(config['train_path'], config['image_dir'], config)
    valid_seen_dataset = LEDDataset(config['train_path'], config['image_dir'], config)
    valid_unseen_dataset = LEDDataset(config['train_path'], config['image_dir'], config)
    print("Created Datasets, Creating DataLoaders")
    train_loader = DataLoader(train_dataset, batch_size=3)
    valid_seen_loader = DataLoader(valid_seen_dataset, batch_size=6)
    valid_unseen_loader = DataLoader(valid_unseen_dataset, batch_size=6)
    print("Created DataLoaders, Instantiating Model")
    led_clip = LEDModel(config)
    print("Instantiated Model, Configuring Training Parameters")
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(led_clip.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    scaler = torch.cuda.amp.GradScaler()
    print("Strating Training")
    training_loop(train_loader, valid_seen_loader, valid_unseen_loader, 20, led_clip, loss_fn, optimizer, scaler, scheduler, config) 


main()