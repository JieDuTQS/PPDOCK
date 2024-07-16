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

accelerator.wait_for_everyone()
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")+'_best_model'
logger = Logger(accelerator=accelerator, log_path=f'{args.resultFolder}/{timestamp}.log')
torch.multiprocessing.set_sharing_strategy('file_system')
train, valid, test= get_data(args, logger)
num_workers = 0
train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=True, pin_memory=False, num_workers=num_workers)
valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=1, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)

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
output_last_epoch_dir = f"best_model/epoch_358"
if os.path.exists(output_last_epoch_dir) and os.path.exists(os.path.join(output_last_epoch_dir, "model.safetensors")):
    accelerator.load_state(output_last_epoch_dir)

if args.dis_map_criterion == 'MSE':
    dis_map_criterion = nn.MSELoss()
    pred_dis = True
if args.compound_coord_loss_function == 'SmoothL1':
    compound_coord_criterion = nn.SmoothL1Loss()

model.eval()
logger.log_message(f"Begin test")
if accelerator.is_main_process:
    metrics = evaluate_model(accelerator, args, test_loader, model, compound_coord_criterion, dis_map_criterion, accelerator.device)
    logger.log_stats(metrics, 1, args, prefix="Test")
accelerator.wait_for_everyone()
