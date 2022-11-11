###########################################################################
# Created by: YI ZHENG
# Email: yizheng@bu.edu
# Copyright (c) 2020
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=4, help='classification classes')
        parser.add_argument('--num_fold', type=int, default=5, help='num of k-fold validation')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--all_stage_0_set', type=str, help='train')
        parser.add_argument('--all_stage_1_set', type=str, help='train')
        parser.add_argument('--all_stage_2_set', type=str, help='train')
        parser.add_argument('--all_stage_3_set', type=str, help='train')
        parser.add_argument('--model_path', type=str, help='path to trained model')
        parser.add_argument('--log_path', type=str, help='path to log files')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--train', action='store_true', default=False, help='train only')
        parser.add_argument('--test', action='store_true', default=False, help='test only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--log_interval_local', type=int, default=10, help='classification classes')
        parser.add_argument('--resume', type=str, default="", help='path for model')
        parser.add_argument('--graphcam', action='store_true', default=False, help='GraphCAM')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr

        args.num_epochs = 120
        args.lr = 1e-4     # COAD: 1e-4; BRCA:        

        # if args.test:
        #     args.num_epochs = 1
        return args
