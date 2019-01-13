"""
	Group Activity Recognition
"""
import argparse
import Runtime
import torch
import Pre
torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='/home/ubuntu/GAR/dataset/', help='Please set the root folder of datasets')
parser.add_argument('--dataset_name', type=str, default='VD', choices=['VD', 'CAD'], help='Please choose one of the dataset')
parser.add_argument('--mode', type=str, default='extract_action_feas', choices=['trainval_action', 'extract_action_feas', 'frame_trainval_activity', 'trainval_activity'], help='Please choose mode')

opt = parser.parse_args()

# Step Zero: Dataset Preprocessing
# 0.0 semantic
#Pre.Processing(opt.dataset_root, opt.dataset_name, operation='Semantic')

# 0.1 vision
#data_info = Dataset_Preprocessing(opt.dataset_root, opt.dataset_name)
#print data_info


# Step One. ***In Service!!!***
Action = Runtime.Action_Level(opt.dataset_root, opt.dataset_name, opt.mode)
#Action.trainval()
Action.extractFeas()


# Step Two
#Activity = Runtime.Activity_Level(opt.dataset_root, opt.dataset_name, opt.mode)
#Activity.trainval()
#Activity.test()

# Step Three, Semantic
#Semantic = Runtime.Semantic_Level(opt.dataset_root, opt.dataset_name, opt.mode)
#Semantic.trainval()