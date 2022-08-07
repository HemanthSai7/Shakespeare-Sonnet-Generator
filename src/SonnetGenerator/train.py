import os
import math
import random

import torch
import numpy as np
from transformers import SubsetRandomSampler
from torch.utils.data import DataLoader
from dataloader import SonnetDataset
from engine import Engine


def perform_run(config,model,tokenizer,weight_path='./',load_weights_path=None):
    
    sonnet_files=['Sonnets.txt']
    datasett=SonnetDataset(sonnet_files,tokenizer)
    indices=list(range(len(datasett)))
    random.shuffle(indices)
    
    split=math.floor(0.3*len(datasett))
    train_indices,val_indices=indices[split:],indices[:split]
    
    train_sampler=SubsetRandomSampler(train_indices)
    val_sampler=SubsetRandomSampler(val_indices)
    
    train_loader=DataLoader(datasett,batch_size=config.batch_size,
                           sampler=train_sampler,num_workers=config.num_workers)
    
    val_loader=DataLoader(datasett,batch_size=config.batch_size,
                           sampler=train_sampler,num_workers=config.num_workers)
    if load_weights_path is not None:
        model.load_state_dict(torch.load(load_weights_path+f"{config.save_file_name}.pt")["model_state_dict"])
        print("Weight loaded")
        
    engine=Engine(model=model.to(config.device),device=config.device,
                config=config,save_file_name=config.save_file_name,
                weight_path=weight_path)
    
    engine.fit(train_loader,val_loader)
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True   
    