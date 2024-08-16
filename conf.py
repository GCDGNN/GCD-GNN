from argparse import ArgumentParser
import os.path as osp
from utils.utils import *



parser = ArgumentParser()
parser.add_argument('--dataset',          type = str,             default = 'tfinance') 
parser.add_argument('--num_workers',   default = 8,                  type = int, choices = [0,8])
parser.add_argument('--seed',          default = 1234,               type = int)
parser.add_argument('--data_dir',         type = str,             default = "datasets/") 
parser.add_argument('--hyper_file',       type = str,             default = 'config/')
parser.add_argument('--log_dir',          type = str,             default = 'logs/')
parser.add_argument('--best_model_path',  type = str,             default = 'checkpoints/')
parser.add_argument('--train_size',       type = float,           default = 0.4)
parser.add_argument('--val_size',         type = float,           default = 0.2)
parser.add_argument('--no_dev',         action = "store_true" ,   default = False)
parser.add_argument('--gpu_id',           type = int,             default = 0)
parser.add_argument('--multirun',         type = int,             default = 1)
parser.add_argument('--model',            type = str,             default ='GCDGNN')
parser.add_argument('--run_best',       action ='store_true',     default = False)
parser.add_argument('--add_relations',       action ='store_true',     default = False)
parser.add_argument('--train_auc', action='store_true', default=False)
parser.add_argument('--pre_load_ckpt',    type=str,               default = None)
parser.add_argument('--load_model_path',    type=str,               default = 'GCDGNN')
parser.add_argument('--test_each_epoch', action= 'store_true',  default=False)





args = parser.parse_args()

config_path = osp.join(args.hyper_file, args.dataset + '.yml')
config = get_config(config_path)
model_name = args.model
config = config[model_name] 
config['model_name'] = model_name
config = args2config(args, config)