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
parser.add_argument("--batch_size", type=int, default=8,help="batch size.")
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

accelerator.wait_for_everyone()

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")+'_best_model'
logger = Logger(accelerator=accelerator, log_path=f'{args.resultFolder}/{timestamp}.log')

torch.multiprocessing.set_sharing_strategy('file_system')

train, valid, test= get_data(args, logger)

logger.log_message(f"data size: test: {len(test)}")

num_workers = 0

train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=True, pin_memory=False, num_workers=num_workers)
valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)

from models.model import *
model = get_model(args)

if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
last_epoch = -1
steps_per_epoch = len(train_loader)
total_training_steps = args.total_epochs * len(train_loader)

scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_training_steps, last_epoch=last_epoch)

(model,optimizer,scheduler,train_loader) = accelerator.prepare(model, optimizer, scheduler, train_loader)

output_last_epoch_dir = 'best_model/epoch_245'
if os.path.exists(output_last_epoch_dir) and os.path.exists(os.path.join(output_last_epoch_dir, "model.safetensors")):
    accelerator.load_state(output_last_epoch_dir)

if args.location_dis_map_criterion == 'MSE':
    location_dis_map_criterion = nn.MSELoss()
    pred_dis = True

if args.location_compound_center_criterion == 'MSE':
    location_compound_center_criterion = nn.MSELoss()

model.eval()
logger.log_message(f"Begin test")
if accelerator.is_main_process:
    metrics = evaluate_model(accelerator, args, test_loader, model, location_dis_map_criterion, location_compound_center_criterion,  accelerator.device)
    logger.log_stats(metrics, 1, args, prefix="Test")
accelerator.wait_for_everyone()


get_compound_center_pred(accelerator, args, train_loader,valid_loader,test_loader, model, location_dis_map_criterion, location_compound_center_criterion,  accelerator.device)