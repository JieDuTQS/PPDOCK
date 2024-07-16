import numpy as np
import os
import torch
import random
from numbers import Number
from accelerate.logging import get_logger
import logging
from tqdm.auto import tqdm
from torch_scatter import scatter_mean
import csv
import pandas as pd

class Logger:
    def __init__(self, accelerator, log_path):
        self.logger = get_logger('Main')
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO)
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
def evaluate_model(accelerator=None, args=None, data_loader=None, model=None, compound_coord_criterion=None, dis_map_criterion=None,device=None):
    dis_map_loss_ppe_r_all=0.0
    dis_map_loss_pc_r_all=0.0
    dis_map_loss_pc_ppe_all=0.0
    compound_coord_loss_all=0.0
    dis_map_length_all=0.0
    compound_coord_length_all=0.0
    loss_epoch=0.0
    rmsd_list=[]
    rmsd_2A_list=[]
    rmsd_5A_list=[]
    compound_centroid_dis_list=[]
    compound_centroid_dis_2A_list=[]
    compound_centroid_dis_5A_list=[] 

    data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for data in data_iter:
        data = data.to(device)
        compound_coord_pred, compound_batch, dis_map_pred_by_pair_embeddings, dis_map_pred_by_coord, dis_map= model(data) 
        compound_coord = data.compound_coords
        dis_map_loss_ppe_r = args.dis_map_loss_pr * dis_map_criterion(dis_map_pred_by_pair_embeddings, dis_map) if len(dis_map) > 0 else torch.tensor([0])
        dis_map_loss_pc_r = args.dis_map_loss_pr * dis_map_criterion(dis_map_pred_by_coord, dis_map) if len(dis_map) > 0 else torch.tensor([0])
        dis_map_loss_pc_ppe = args.dis_map_loss_pp * dis_map_criterion(dis_map_pred_by_coord, dis_map_pred_by_pair_embeddings) if len(dis_map_pred_by_pair_embeddings) > 0 else torch.tensor([0])
        compound_coord_loss = args.compound_coord_loss_weight * compound_coord_criterion(compound_coord_pred, compound_coord) if len(compound_coord) > 0 else torch.tensor([0])
        sd = ((compound_coord_pred.detach() - compound_coord) ** 2).sum(dim=-1) 
        rmsd = scatter_mean(sd, index=compound_batch, dim=0).sqrt().detach() 
        compound_centroid_pred = scatter_mean(src=compound_coord_pred, index=compound_batch, dim=0) 
        compound_centroid_true = scatter_mean(src=compound_coord, index=compound_batch, dim=0)
        compound_centroid_dis = (compound_centroid_pred - compound_centroid_true).norm(dim=-1) 
        dis_map_loss_ppe_r_all+=len(dis_map_pred_by_pair_embeddings)*dis_map_loss_ppe_r.item()
        dis_map_loss_pc_r_all+=len(dis_map_pred_by_coord)*dis_map_loss_pc_r.item()
        dis_map_loss_pc_ppe_all+=len(dis_map_pred_by_coord)*dis_map_loss_pc_ppe.item()
        dis_map_length_all+=len(dis_map)
        compound_coord_loss_all+=len(compound_coord_pred)*compound_coord_loss.item()
        compound_coord_length_all+=len(compound_coord_pred)
        rmsd_list.append(rmsd.detach()) 
        rmsd_2A_list.append((rmsd.detach() < 2).float()) 
        rmsd_5A_list.append((rmsd.detach() < 5).float()) 
        compound_centroid_dis_list.append(compound_centroid_dis.detach()) 
        compound_centroid_dis_2A_list.append((compound_centroid_dis.detach() < 2).float()) 
        compound_centroid_dis_5A_list.append((compound_centroid_dis.detach() < 5).float())      
    loss_epoch = (compound_coord_loss_all/compound_coord_length_all)+\
                    (dis_map_loss_ppe_r_all/dis_map_length_all)+\
                    (dis_map_loss_pc_r_all/dis_map_length_all)+\
                    (dis_map_loss_pc_ppe_all/dis_map_length_all)
    rmsd = torch.cat(rmsd_list)
    rmsd_2A = torch.cat(rmsd_2A_list)
    rmsd_5A = torch.cat(rmsd_5A_list)
    rmsd_25 = torch.quantile(rmsd, 0.25)
    rmsd_50 = torch.quantile(rmsd, 0.50)
    rmsd_75 = torch.quantile(rmsd, 0.75)
    compound_centroid_dis = torch.cat(compound_centroid_dis_list)
    compound_centroid_dis_2A = torch.cat(compound_centroid_dis_2A_list)
    compound_centroid_dis_5A = torch.cat(compound_centroid_dis_5A_list)
    compound_centroid_dis_25 = torch.quantile(compound_centroid_dis, 0.25)
    compound_centroid_dis_50 = torch.quantile(compound_centroid_dis, 0.50)
    compound_centroid_dis_75 = torch.quantile(compound_centroid_dis, 0.75)
    metrics = {}
    metrics.update({"dis_map_loss_ppe_r":dis_map_loss_ppe_r_all/dis_map_length_all, "dis_map_loss_pc_r":dis_map_loss_pc_r_all/dis_map_length_all, "dis_map_loss_pc_ppe": dis_map_loss_pc_ppe_all/dis_map_length_all})
    metrics.update({"compound_coord_loss": compound_coord_loss_all/compound_coord_length_all}) 
    metrics.update({"loss_epoch": loss_epoch})
    metrics.update({"rmsd": rmsd.mean().item(), "rmsd < 2A": rmsd_2A.mean().item(), "rmsd < 5A": rmsd_5A.mean().item()})
    metrics.update({"rmsd 25%": rmsd_25.item(), "rmsd 50%": rmsd_50.item(), "rmsd 75%": rmsd_75.item()})
    metrics.update({"centroid_dis": compound_centroid_dis.mean().item(), "centroid_dis < 2A": compound_centroid_dis_2A.mean().item(), "centroid_dis < 5A": compound_centroid_dis_5A.mean().item()})
    metrics.update({"centroid_dis 25%": compound_centroid_dis_25.item(), "centroid_dis 50%": compound_centroid_dis_50.item(), "centroid_dis 75%": compound_centroid_dis_75.item()})
    return metrics