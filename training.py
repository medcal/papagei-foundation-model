# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch 
import os
import dataset
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import joblib
import wandb
from datetime import datetime
from models import cnn, efficientnet, transformer
from pytorch_metric_learning import losses
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

def train_step(model, dataloader, criterion, optimizer, device):

    """
    One training epoch for a SimCLR model

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion (torch.nn.<Loss>): Loss function to optimizer
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU

    Returns:
        train_loss (float): The training loss for the epoch
    """
    
    model.to(device)
    model.train()
    train_loss = 0
    for i, (X, y) in enumerate(dataloader):
        signal_view1 = X[0].to(device)
        signal_view2 = X[1].to(device)

        z_1, z_2 = model(signal_view1), model(signal_view2)
        loss = criterion(z_1, z_2)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)

def training(model, epochs, train_dataloader, criterion, optimizer, device, wandb=None):

    """
    Training a SimCLR model

    Args:
        model (torch.nn.Module): Model to train
        epochs (int): No. of epochs to train
        train_dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion (torch.nn.<Loss>): Loss function to optimizer
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        wandb (wandb): wandb object for experiment tracking

    Returns:
        dict_log (dictionary): A dictionary log with metrics
    """

    dict_log = {'train_loss': []}
    
    for e in tqdm(range(epochs)):
        epoch_loss = train_step(model=model,
                               dataloader=train_dataloader,
                               criterion=criterion,
                               optimizer=optimizer,
                               device=device)
        if wandb:
            wandb.log({"Train Loss": epoch_loss})

        dict_log['train_loss'].append(epoch_loss)
        print(f"Epoch: {e+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

    return dict_log

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    df_train = pd.read_csv("../data/vitaldbmeta/train.csv")
    df_val = pd.read_csv("../data/vitaldbmeta/val.csv")
    df_test = pd.read_csv("../data/vitaldbmeta/test.csv")

    prob_dictionary = {'g_p': 0.3, 'n_p': 0.20, 'w_p':0.0, 'f_p':0.20, 's_p':0.25, 'c_p':0.5}
    batch_size = 16
    num_workers = 0
    normalization = True
    path = "../data/vitaldbppg/"
    label = "sex" # does not matter for SSL

    train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloader(df_train=df_train,
                                                                            df_val=df_val,
                                                                            df_test=df_test,
                                                                            path=path,
                                                                            label_name=label,
                                                                            batch_size=batch_size,
                                                                            prob_dictionary=prob_dictionary,
                                                                            normalization=normalization,
                                                                            num_workers=num_workers)
    
    # model_config = {'d_model': 5000,
    #            'nhead': 2,
    #            'dim_feedforward': 2048,
    #            'trans_dropout': 0.0,
    #            'proj_dropout': 0.0,
    #            'num_layers': 2,
    #            'h1': 1024,
    #            'embedding_size': 512}
    # model = transformer.TransformerSimple(model_config=model_config)


    model_config = {'h1': 64,
                    'h2': 32,
                    'h3': 128,
                    'h4': 256,
                    'h5': 384,
                    'h6': 512,
                    'h7': 768,
                    'h8': 1024}

    model = efficientnet.EfficientNetB0Base(in_channels=1, dict_channels=model_config)
    epochs = 3000
    lr = 0.0001
    criterion = losses.SelfSupervisedLoss(losses.NTXentLoss())
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    device = "cuda:5" if torch.cuda.is_available() else "cpu"

    ### Experiment Traking ###
    experiment_name = "EfficientNet"
    group_name = "PPG"

    config = {"learning_rate": lr, 
         "epochs": epochs,
         "batch_size": batch_size,
         "augmentations": prob_dictionary}

    wandb.init(project=experiment_name,
            config=config | model_config, 
            name=experiment_name,
            group=group_name)

    dict_log = training(model=model, 
                   train_dataloader=train_dataloader,
                   epochs=epochs,
                   criterion=criterion,
                   optimizer=optimizer,
                   device=device,
                   wandb=wandb)
    
    run_id = wandb.run.id
    time = datetime.now().strftime(("%Y-%m-%d-%H-%M-%S"))
    model_filename = f'{experiment_name}_{run_id}_{time}'
    joblib.dump(dict_log, "../models/"+ model_filename +"_log.p")
    torch.save(model.state_dict(), "../models/" + model_filename + ".pt")
    wandb.finish()