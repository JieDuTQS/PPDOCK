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
parser.add_argument('--interactionThresholdDistance', type=float, default=20.0,help="Set the maximum value of the distance map between pocket and ligand")
parser.add_argument("--batch_size", type=int, default=6,help="batch size.")
parser.add_argument('--optim', type=str, default='adam', choices=['adam'])
parser.add_argument("--lr", type=float, default=5e-05,help="learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--total_epochs", type=int, default=500,help="total training epochs.")
parser.add_argument("--tqdm_interval", type=float, default=10,help="tqdm bar update interval")
parser.add_argument('--clip_grad', action='store_true', default=True)
parser.add_argument('--log-interval', type=int, default=100)


parser.add_argument('--coordinate_scale', type=float, default=5.0)
parser.add_argument("--protein_feat_size", type=int, default=1280)
parser.add_argument("--compound_feat_size", type=int, default=56)

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


parser.add_argument('--complex_whole_protein_rdkit_coords_translation_noise_mode', type=str, default='random', choices=['None','random'])
parser.add_argument('--complex_whole_protein_rdkit_coords_noise_translation_random_max_dist', type=float, default=3.0)
parser.add_argument("--location_dis_map_criterion", type=str, default='MSE', choices=['MSE'])
parser.add_argument("--location_compound_center_criterion", type=str, default='MSE', choices=['MSE'])
parser.add_argument("--location_dis_map_loss_weight", type=float, default=1.0)
parser.add_argument("--location_compound_center_loss_weight", type=float, default=1.0)
parser.add_argument("--location_hidden_size", type=int, default=128)
parser.add_argument('--location_num_layers', type=int, default=2)
parser.add_argument('--location_num_iters', type=int, default=3)

parser.add_argument('--location_pred_type', type=str, default="EGNN_pointnet", choices=['EGNN_mean', 'pointnet_EGNN', 'EGNN_pointnet'])
parser.add_argument('--compound_global_feat_size', type=int, default=1024)
args = parser.parse_args()

if args.location_pred_type=='pointnet_EGNN':
    args.rm_LAS_constrained_optim=True
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
if args.location_pred_type=='pointnet_EGNN' or args.location_pred_type== 'EGNN_pointnet':
    DataLoader_drop_last= True
else:
    DataLoader_drop_last= False
train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=True, pin_memory=False, num_workers=num_workers,drop_last=DataLoader_drop_last)
valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers,drop_last=DataLoader_drop_last)
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers,drop_last=DataLoader_drop_last)

from models.model import *
device = 'cuda'

model = get_model(args)

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
    
if args.location_dis_map_criterion == 'MSE':
    location_dis_map_criterion = nn.MSELoss()
    pred_dis = True

if args.location_compound_center_criterion == 'MSE':
    location_compound_center_criterion = nn.MSELoss()
    
logger.log_message(f"Total epochs: {args.total_epochs}")
logger.log_message(f"Total training steps: {total_training_steps}")

for epoch in range(last_epoch+1, args.total_epochs):
    model.train()

    location_dis_map_loss_all=0.0
    location_compound_center_loss_all=0.0
    dis_map_length_all=0.0
    compound_coord_length_all=0.0
    loss_epoch=0.0
    

    compound_center_dis_list=[]
    compound_center_dis_2A_list=[]
    compound_center_dis_5A_list=[] 
    
    data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for batch_id, data in enumerate(data_iter, start=1):
        optimizer.zero_grad()

        compound_center_pred,compound_center_true,protein_com_center_dis_map_pred,protein_com_center_dis_map_true,compound_center_pred_pointnet= model(data)
        if  compound_center_pred.isnan().any() or compound_center_true.isnan().any() or protein_com_center_dis_map_pred.isnan().any() or protein_com_center_dis_map_true.isnan().any():
            print(f"nan occurs in epoch {epoch}")
            continue

        location_dis_map_loss = args.location_dis_map_loss_weight * location_dis_map_criterion(protein_com_center_dis_map_pred, protein_com_center_dis_map_true) if len(protein_com_center_dis_map_true) > 0 else torch.tensor([0])

        location_compound_center_loss = args.location_compound_center_loss_weight * location_compound_center_criterion(compound_center_pred, compound_center_true) if len(compound_center_true) > 0 else torch.tensor([0])

        location_compound_center_pred_pointnet_loss= args.location_compound_center_loss_weight * location_compound_center_criterion(compound_center_pred_pointnet, compound_center_true) if len(compound_center_pred_pointnet) > 0 else torch.tensor([0])
        compound_center_dis = (compound_center_pred - compound_center_true).norm(dim=-1).squeeze(-1) 
        
        loss = location_dis_map_loss + location_compound_center_loss 
            
        accelerator.backward(loss)
        if args.clip_grad: 
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
        optimizer.step()
        scheduler.step()
        
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

    
    logger.log_stats(metrics, epoch, args, prefix="Train")
    if accelerator.is_main_process:
        metrics_runtime_no_prefix(metrics, train_writer, epoch) 
    
    accelerator.wait_for_everyone()
    
    compound_center_dis, compound_center_dis_2A, compound_center_dis_5A = None, None, None
    
    model.eval()
    logger.log_message(f"Begin validation")
    if accelerator.is_main_process:
        metrics = evaluate_model(accelerator, args, valid_loader, model, location_dis_map_criterion, location_compound_center_criterion,  accelerator.device)
        logger.log_stats(metrics, epoch, args, prefix="Valid")
        metrics_runtime_no_prefix(metrics, valid_writer, epoch)
    
    logger.log_message(f"Begin test")
    if accelerator.is_main_process:
        metrics = evaluate_model(accelerator, args, test_loader, model, location_dis_map_criterion, location_compound_center_criterion,  accelerator.device)
        logger.log_stats(metrics, epoch, args, prefix="Test")
        metrics_runtime_no_prefix(metrics, test_writer, epoch)

        output_dir = f"{pre}/models/epoch_{epoch}"
        accelerator.save_state(output_dir=output_dir)
        accelerator.save_state(output_dir=output_last_epoch_dir)

    accelerator.wait_for_everyone()
