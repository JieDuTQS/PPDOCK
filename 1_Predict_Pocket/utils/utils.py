import numpy as np
import os
import torch
import random
from numbers import Number
from accelerate.logging import get_logger
import logging
from tqdm.auto import tqdm
from torch_scatter import scatter_mean
import pandas as pd
import time

class Logger:
    def __init__(self, accelerator, log_path):
        self.logger = get_logger('Main')
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter('%(message)s', ""))
        self.logger.logger.addHandler(handler)
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is {os.getcwd()}')

    def log_stats(self, stats, epoch, args, prefix=''):
        msg_start = f'[{prefix}] Epoch {epoch} out of {args.total_epochs}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.5f}' for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)

def Seed_everything(seed=36):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def metrics_runtime_no_prefix(metrics, writer, epoch):
    for key in metrics.keys():
        if isinstance(metrics[key], Number):
            writer.add_scalar(f'{key}', metrics[key], epoch)
        elif torch.is_tensor(metrics[key]) and metrics[key].numel() == 1:
            writer.add_scalar(f'{key}', metrics[key].item(), epoch)
            
@torch.no_grad()
def evaluate_model(accelerator=None, args=None, data_loader=None, model=None, location_dis_map_criterion=None,location_compound_center_criterion=None,device=None):
    location_dis_map_loss_all=0.0
    location_compound_center_loss_all=0.0
    dis_map_length_all=0.0
    compound_coord_length_all=0.0
    loss_epoch=0.0
    
    compound_center_dis_list=[]
    compound_center_dis_2A_list=[]
    compound_center_dis_5A_list=[] 

    data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for data in data_iter:
        data = data.to(device)

        compound_center_pred,compound_center_true,protein_com_center_dis_map_pred,protein_com_center_dis_map_true,compound_center_pred_pointnet= model(data) 

        location_dis_map_loss = args.location_dis_map_loss_weight * location_dis_map_criterion(protein_com_center_dis_map_pred, protein_com_center_dis_map_true) if len(protein_com_center_dis_map_true) > 0 else torch.tensor([0])
        location_compound_center_loss = args.location_compound_center_loss_weight * location_compound_center_criterion(compound_center_pred, compound_center_true) if len(compound_center_true) > 0 else torch.tensor([0])
        compound_center_dis = (compound_center_pred - compound_center_true).norm(dim=-1).squeeze(-1)  
            
        location_dis_map_loss_all+=len(protein_com_center_dis_map_pred)*location_dis_map_loss.item()
        dis_map_length_all+=len(protein_com_center_dis_map_pred)
        
        location_compound_center_loss_all+=len(compound_center_pred)*location_compound_center_loss.item()
        compound_coord_length_all+=len(compound_center_pred)
        
        compound_center_dis_list.append(compound_center_dis.detach()) 
        compound_center_dis_2A_list.append((compound_center_dis.detach() < 2).float()) 
        compound_center_dis_5A_list.append((compound_center_dis.detach() < 5).float()) 

    loss_epoch = (location_dis_map_loss_all/dis_map_length_all)+\
                    (location_compound_center_loss_all/compound_coord_length_all)
                    

    compound_center_dis = torch.cat(compound_center_dis_list)
    compound_center_dis_2A = torch.cat(compound_center_dis_2A_list)
    compound_center_dis_5A = torch.cat(compound_center_dis_5A_list)
 
    compound_center_dis_25 = torch.quantile(compound_center_dis, 0.25)
    compound_center_dis_50 = torch.quantile(compound_center_dis, 0.50)
    compound_center_dis_75 = torch.quantile(compound_center_dis, 0.75)
           
    metrics = {}
    metrics.update({"location_dis_map_loss":location_dis_map_loss_all/dis_map_length_all})
    metrics.update({"location_compound_center_loss": location_compound_center_loss_all/compound_coord_length_all}) 
    metrics.update({"loss_epoch": loss_epoch})
    metrics.update({"anchor_dis": compound_center_dis.mean().item(), "anchor_dis < 2A": compound_center_dis_2A.mean().item(), "anchor_dis < 5A": compound_center_dis_5A.mean().item()})
    metrics.update({"anchor_dis 25%": compound_center_dis_25.item(), "anchor_dis 50%": compound_center_dis_50.item(), "anchor_dis 75%": compound_center_dis_75.item()})
    return metrics

@torch.no_grad()
def get_compound_center_pred(accelerator=None, args=None, train_loader=None,valid_loader=None,test_loader=None, model=None, location_dis_map_criterion=None, location_compound_center_criterion=None, device=None):
    dict_test={}
    data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for data in data_iter:
        data = data.to(device)
        compound_center_pred,compound_center_true,protein_com_center_dis_map_pred,protein_com_center_dis_map_true,compound_center_pred_pointnet= model(data) 
        for i in range (compound_center_pred.size()[0]):
            dict_test.update({data['pdb'][i]:[compound_center_pred[i].cpu().numpy(),compound_center_true[i].cpu().numpy()]})
    
    data_iter = tqdm(valid_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for data in data_iter:
        data = data.to(device)
        compound_center_pred,compound_center_true,protein_com_center_dis_map_pred,protein_com_center_dis_map_true,compound_center_pred_pointnet= model(data) 
        for i in range (compound_center_pred.size()[0]):
            dict_test.update({data['pdb'][i]:[compound_center_pred[i].cpu().numpy(),compound_center_true[i].cpu().numpy()]})
    
    data_iter = tqdm(test_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)

    for data in data_iter:
        data = data.to(device)
        compound_center_pred,compound_center_true,protein_com_center_dis_map_pred,protein_com_center_dis_map_true,compound_center_pred_pointnet= model(data) 
        for i in range (compound_center_pred.size()[0]):
            dict_test.update({data['pdb'][i]:[compound_center_pred[i].cpu().numpy(),compound_center_true[i].cpu().numpy()]})            

    df = pd.DataFrame(dict_test)
    torch.save(df,'anchor.pt')
