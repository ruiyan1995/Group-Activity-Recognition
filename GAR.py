"""
	Group Activity Recognition
"""
import argparse
import Runtime
import torch
torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='/home/ubuntu/GAR/dataset/', help='Please set the root folder of datasets')
parser.add_argument('--dataset_name', type=str, default='VD', choices=['VD', 'CAD'], help='Please choose one of the dataset')
parser.add_argument('--mode', type=str, default='trainval_action', choices=['trainval_action', 'extract_action_feas', 'frame_trainval_activity'], help='Please choose one of the dataset')

opt = parser.parse_args()

# Step Zero: Dataset Preprocessing
#data_info = Dataset_Preprocessing(opt.dataset_root, opt.dataset_name)
#print data_info

# Step One
Action = Runtime.Action_Level(opt.dataset_root, opt.dataset_name, opt.mode)
Action.trainval()
#Action.extract_feas(save_folder)


# Step Two
#Activity = ActivityLevel()
#Activity.train()
#Activity.test()