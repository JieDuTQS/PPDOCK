import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
import torch
from torch_geometric.loader import DataLoader
import tqdm
from torch_scatter import scatter_mean
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from data import get_data
from utils.utils import *

parser = argparse.ArgumentParser(description='model training.')
parser.add_argument('--seed', type=int, default=42,help="seed to use.")
parser.add_argument("--mixed_precision", type=str, default='no', choices=['no', 'fp16'])
parser.add_argument("--resultFolder", type=str, default="./result",help="information you want to keep a record.")
parser.add_argument("--exp_name", type=str, default="train",help="data path.")
parser.add_argument("--data_path", type=str, default="pdbbind2020/dataset",help="Data path.")
parser.add_argument("--pocket_mode", type=str, default='My_pocket_mode', choices=['My_pocket_mode','FABind_pocket_mode'])
parser.add_argument("--add_noise", type=bool, default=True)
parser.add_argument("--pocket_obtaining_method", type=str, default='compound_coords_center', choices=['compound_coords_center','dis_map_pro_com'])
parser.add_argument('--add_noise_center', type=float, default=5.0)
parser.add_argument('--pocket_getting_percent', type=float, default=0.15)
parser.add_argument('--pocket_minimum_quantity', type=int, default=20)
parser.add_argument('--pocket_radius_minimum', type=float, default=5.0)
parser.add_argument('--radius_distance_interval', type=float, default=1.0)
parser.add_argument("--add_noise_to_poc_com", type=int, default=1,choices=[0,1])
parser.add_argument('--pocket_radius', type=float, default=10.0)
parser.add_argument('--interactionThresholdDistance', type=float, default=10.0,help="Set the maximum value of the distance map between pocket and ligand")
parser.add_argument("--rdkit_coords_init_mode", type=str, default="pocket_center_rdkit",choices=['pocket_center_rdkit'])
parser.add_argument("--batch_size", type=int, default=8,help="batch size.")
parser.add_argument('--optim', type=str, default='adam', choices=['adam'])
parser.add_argument("--lr", type=float, default=5e-05,help="learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--total_epochs", type=int, default=500,help="total training epochs.")
parser.add_argument("--dis_map_criterion", type=str, default='MSE', choices=['MSE'])
parser.add_argument("--compound_coord_loss_function", type=str, default='SmoothL1', choices=['SmoothL1'])
parser.add_argument("--tqdm_interval", type=float, default=10,help="tqdm bar update interval")
parser.add_argument("--dis_map_loss_pr", type=float, default=1.0)
parser.add_argument("--dis_map_loss_pp", type=float, default=1.0)
parser.add_argument("--compound_coord_loss_weight", type=float, default=1.0)
parser.add_argument('--clip_grad', action='store_true', default=True)
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument("--model_net", type=str, default='FABind_layer', choices=['FABind_layer'])
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument('--coordinate_scale', type=float, default=5.0)
parser.add_argument("--protein_feat_size", type=int, default=1280)
parser.add_argument("--compound_feat_size", type=int, default=56)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_iters', type=int, default=8)
parser.add_argument('--inter_cutoff', type=float, default=10.0,help="Threshold for establishing edges between protein residues and ligand molecules.")
parser.add_argument('--intra_cutoff', type=float, default=8.0,help="The threshold for establishing edges between protein residues.")
parser.add_argument('--random_n_iter', action='store_true', default=True)
parser.add_argument('--geometry_reg_step_size', type=float, default=0.001)
parser.add_argument('--rm_layernorm', action='store_true', default=True)
parser.add_argument('--opm', action='store_true', default=False)
parser.add_argument('--add_cross_attn_layer', action='store_true', default=True)
parser.add_argument('--explicit_pair_embed', action='store_true', default=True)
parser.add_argument('--keep_trig_attn', action='store_true', default=False)
parser.add_argument('--add_attn_pair_bias', action='store_true', default=True)
parser.add_argument('--rm_F_norm', action='store_true', default=False)
parser.add_argument('--norm_type', type=str, default="per_sample", choices=['per_sample', '4_sample', 'all_sample'])
parser.add_argument('--fix_pocket', action='store_true', default=False)
parser.add_argument('--rm_LAS_constrained_optim', action='store_true', default=False)
args = parser.parse_args()

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision) 
set_seed(args.seed)
pre = f"{args.resultFolder}/{args.exp_name}" 
if accelerator.is_main_process:
    os.system(f"mkdir -p {pre}/models")
    tensorboard_dir = f"{pre}/tensorboard"
    os.system(f"mkdir -p {tensorboard_dir}")
    train_writer = SummaryWriter(log_dir=f'{tensorboard_dir}/train')
    valid_writer = SummaryWriter(log_dir=f'{tensorboard_dir}/valid')
    test_writer = SummaryWriter(log_dir=f'{tensorboard_dir}/test')
accelerator.wait_for_everyone()
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
logger = Logger(accelerator=accelerator, log_path=f'{pre}/{timestamp}.log')
torch.multiprocessing.set_sharing_strategy('file_system')
train, valid, test= get_data(args, logger)
logger.log_message(f"data size: train: {len(train)}, valid: {len(valid)}, test: {len(test)}")
num_workers = 0
train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=True, pin_memory=False, num_workers=num_workers)
valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
from models.model import *
device = 'cuda'
model = get_model(args, logger, device)
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
last_epoch = -1
steps_per_epoch = len(train_loader)
total_training_steps = args.total_epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_training_steps, last_epoch=last_epoch)
(model,optimizer,scheduler,train_loader) = accelerator.prepare(model, optimizer, scheduler, train_loader)
output_last_epoch_dir = f"{pre}/models/epoch_last"
if os.path.exists(output_last_epoch_dir) and os.path.exists(os.path.join(output_last_epoch_dir, "model.safetensors")):
    accelerator.load_state(output_last_epoch_dir)
    last_epoch = round(scheduler.state_dict()['last_epoch'] / steps_per_epoch) - 1
    logger.log_message(f'Load model from epoch: {last_epoch}')
if args.dis_map_criterion == 'MSE':
    dis_map_criterion = nn.MSELoss()
    pred_dis = True
if args.compound_coord_loss_function == 'SmoothL1':
    compound_coord_criterion = nn.SmoothL1Loss()
logger.log_message(f"Total epochs: {args.total_epochs}")
logger.log_message(f"Total training steps: {total_training_steps}")
for epoch in range(last_epoch+1, args.total_epochs):
    model.train()
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
    data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process) 
    for batch_id, data in enumerate(data_iter, start=1):
        # if batch_id>10:
        #     break
        optimizer.zero_grad()
        compound_coord_pred, compound_batch, dis_map_pred_by_pair_embeddings, dis_map_pred_by_coord, dis_map= model(data)
        if compound_coord_pred.isnan().any() or dis_map_pred_by_pair_embeddings.isnan().any() or dis_map_pred_by_coord.isnan().any():
            print(f"nan occurs in epoch {epoch}")
            continue
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
        loss = compound_coord_loss + \
            dis_map_loss_ppe_r + dis_map_loss_pc_r + dis_map_loss_pc_ppe
        accelerator.backward(loss)
        if args.clip_grad: 
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()
        scheduler.step()
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
    logger.log_stats(metrics, epoch, args, prefix="Train")
    if accelerator.is_main_process:
        metrics_runtime_no_prefix(metrics, train_writer, epoch) 
    accelerator.wait_for_everyone()
    rmsd, rmsd_2A, rmsd_5A = None, None, None
    compound_centroid_dis, compound_centroid_dis_2A, compound_centroid_dis_5A = None, None, None
    model.eval()
    logger.log_message(f"Begin validation")
    if accelerator.is_main_process:
        metrics = evaluate_model(accelerator, args, valid_loader, model, compound_coord_criterion, dis_map_criterion, accelerator.device)
        logger.log_stats(metrics, epoch, args, prefix="Valid")
        metrics_runtime_no_prefix(metrics, valid_writer, epoch)
    logger.log_message(f"Begin test")
    if accelerator.is_main_process:
        metrics = evaluate_model(accelerator, args, test_loader, model, compound_coord_criterion, dis_map_criterion, accelerator.device)
        logger.log_stats(metrics, epoch, args, prefix="Test")
        metrics_runtime_no_prefix(metrics, test_writer, epoch)
        output_dir = f"{pre}/models/epoch_{epoch}"
        accelerator.save_state(output_dir=output_dir)
        accelerator.save_state(output_dir=output_last_epoch_dir)
    accelerator.wait_for_everyone()
