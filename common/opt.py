import argparse
from email.policy import default
import os
import math
import time
import torch

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        # model args
        self.parser.add_argument('--shhpe_model', type=str, default='htnet')      
        
        # train args
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--nepoch', type=int, default=30)
        self.parser.add_argument('--batch_size', type=int, default=512)
        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('--lr', type=float, default=0.0005)
        self.parser.add_argument('--large_decay_epoch', type=int, default=5)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)        
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)     
        self.parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help ='lower lr bound for cyclic schedulers that hit 0')   
        self.parser.add_argument('--workers', type=int, default=4)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--drop',default=0.2, type=float)
        self.parser.add_argument('--seed',default=1, type=int)        
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)        
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--root_path', type=str, default='./dataset/',help='Put the dataset into this file')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)  
        self.parser.add_argument('--stride', default=1, type=float)       
        self.parser.add_argument('--lr_min',type=float,default=0,help='Min learn rate') 
        self.parser.add_argument('--crop_uv', type=int, default=0)

        self.parser.add_argument('--save_ckpt_path', type=str, default='./ckpt/your_model')
        
        
        # mutil-hypothesis
        self.parser.add_argument('--mutilhyp_test', action='store_true')
        self.parser.add_argument('--reload_sigma', action='store_true')
        self.parser.add_argument('--sample_nums', type=int, default=10)
        self.parser.add_argument('--alpha', type=float, default=None)



    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        self.opt.pad = 0
        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'

        if self.opt.dataset == 'h36m':
            self.opt.subjects_train = 'S1,S5,S6,S7,S8'
            self.opt.subjects_test = 'S9,S11'

            self.opt.n_joints = 17
            self.opt.out_joints = 17

            self.opt.joints_left = [4, 5, 6, 11, 12, 13] 
            self.opt.joints_right = [1, 2, 3, 14, 15, 16]

            if self.opt.alpha is None:
                self.opt.alpha=0.005

        elif self.opt.dataset.startswith('3dhp'):
            self.opt.subjects_train = 'S1,S2,S3,S4,S5,S6,S7,S8' 
            self.opt.subjects_test = 'TS1,TS2,TS3,TS4,TS5,TS6' # all
            # self.opt.subjects_test = 'TS1,TS2' # GS
            # self.opt.subjects_test = 'TS3,TS4' # no GS
            # self.opt.subjects_test = 'TS5,TS6' # Outdoor

            self.opt.n_joints, self.opt.out_joints = 17, 17
            self.opt.joints_left, self.opt.joints_right = [4,5,6,11,12,13], [1,2,3,14,15,16]

            if self.opt.alpha is None:
                self.opt.alpha=0.01


        if self.opt.train:
            
            if not os.path.exists(self.opt.save_ckpt_path):
                os.makedirs(self.opt.save_ckpt_path)


            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
            file_name = os.path.join(self.opt.save_ckpt_path, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')

        return self.opt






