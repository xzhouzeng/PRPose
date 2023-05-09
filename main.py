import copy
import glob
import logging
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

import common.eval_cal as eval_cal
from common.h36m_dataset import Human36mDataset
from common.load_data_3dhp import Fusion_3dhp
from common.load_data_hm36 import Fusion
from common.mpi_inf_3dhp_dataset import Mpi_inf_3dhp_Dataset
from common.opt import opts
from common.utils import *
from models.avg_model.mlp import LinearModel
from models.avg_model.modulated_gcn import ModulatedGCN
from models.htnet.GCN_conv import adj_mx_from_skeleton
from models.htnet.trans import HTNet
from models.mgcn_lw.modulated_gcn import MGCN_lw

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

def test(args,actions, dataloader, sh_model,avg_model):

    if args.mutilhyp_test:
        print(f"INFO: Generate multiple hypotheses with a number of {args.sample_nums}")
    else:
        print("INFO: Generate without multiple hypotheses ")
        
    sample_nums=args.sample_nums

    avg_model.eval()
    sh_model.eval()

    action_error = define_error_list(actions)

    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe('test', [input_2D, gt_3D, batch_cam, scale, bb_box])

        N = input_2D.size(0)

        if args.mutilhyp_test:

            sigma=avg_model(input_2D.view(-1,1,17,2)).view(N,2,1,17,1).repeat(1,1,1,1,2) # x and y have the same adaptive variance

            sigma=torch.clamp(sigma,min=1)*args.alpha
                
            random_scalse=(torch.randn((N*sample_nums,2,1,17,2),device=sigma.device)*(sigma.repeat(sample_nums,1,1,1,1)))

            input_2D=input_2D.repeat(sample_nums,1,1,1,1)+random_scalse

            output_3D_whole=sh_model(input_2D.view(N*sample_nums*2,1,17,2)).view(N*sample_nums,2,1,17,3)

            output_3D_non_flip = output_3D_whole[:, 0].view(sample_nums,N,17,3).permute(1,0,2,3)
            output_3D_flip     = output_3D_whole[:, 1].view(sample_nums,N,17,3).permute(1,0,2,3)

        else:
            # no sample
            output_3D_whole=sh_model(input_2D.view(N*2,1,17,2)).view(N,2,1,17,3)
            output_3D_non_flip = output_3D_whole[:, 0]
            output_3D_flip     = output_3D_whole[:, 1]


        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, args.joints_left + args.joints_right, :] = output_3D_flip[:, :, args.joints_right + args.joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        out_target = gt_3D.clone()
        out_target = out_target[:, args.pad].unsqueeze(1)

        output_3D[:, :, 0] = 0
        out_target[:, :, 0] = 0

        action_error = eval_cal.test_calculation_muyilhyp(output_3D, out_target, action, action_error, args.dataset, subject)
    
    p1, p2, pck = print_error(args.dataset, action_error)

    return p1, p2, pck

class SigmaDataset(torch.utils.data.Dataset):

    def __init__(self, input_2D_list,gt_3D_list,pred_3D_list,distance_targe):
       self.input_2D_list=input_2D_list
       self.gt_3D_list=gt_3D_list
       self.pred_3D_list=pred_3D_list
       self.distance_targe=distance_targe

    def __len__(self):
        
        return self.input_2D_list.shape[0]

    def __getitem__(self, index):
        
        return self.input_2D_list[index], self.gt_3D_list[index],self.pred_3D_list[index],self.distance_targe[index]

def get_sigma_dataloader(opt,model,dataLoader,model_name,type='train'):
    # sigmadataset v1
    model.eval()
    sigmadata_path=f'sigma_{type}set_{model_name}.pt'
    if os.path.exists(sigmadata_path):
        input_2D_list,gt_3D_list,pred_3D_list=torch.load(sigmadata_path)
    else:
        input_2D_list=[]
        gt_3D_list=[]
        pred_3D_list=[]
        # action_list=[]
        print("Building a dataset of sigma......")
        for i, data in enumerate(tqdm(dataLoader, 0)):

            batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
            input_2D_list.append(input_2D)
            gt_3D_list.append(gt_3D)
            # action_list.append(action)
            [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe('train', [input_2D, gt_3D, batch_cam, scale, bb_box])

            with torch.no_grad():
                output_3D = model(input_2D)
            pred_3D_list.append(output_3D.cpu())
            
        input_2D_list=torch.cat(input_2D_list,dim=0)
        gt_3D_list=torch.cat(gt_3D_list,dim=0)
        pred_3D_list=torch.cat(pred_3D_list,dim=0)
        # action_list=torch.cat(action_list,dim=0)
        
        torch.save((input_2D_list,gt_3D_list,pred_3D_list),sigmadata_path)
    print('input_2D_list.shape:   ',input_2D_list.shape)
    
    gt_3D_list_targe=gt_3D_list.clone()
    gt_3D_list_targe[:,:,0]=0

    distance=torch.norm(gt_3D_list_targe - pred_3D_list, dim=-1,keepdim=True)

    distance_mean=torch.mean(distance)
    distance_targe=distance/distance_mean
    sigma_dataset=SigmaDataset(input_2D_list,gt_3D_list,pred_3D_list,distance_targe)
    sigma_dataloader=torch.utils.data.DataLoader(sigma_dataset, batch_size=opt.batch_size,
                                                shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    return sigma_dataloader

    
def train(opt, actions, avg_model, lr, optimizer, sigma_train_dataloader):
    L2= torch.nn.MSELoss()
    print("Start to train avg model....")
    for epoch in range(1, opt.nepoch):
        loss_all = {'loss': AccumLoss()}

        avg_model.train()
        
        for i, data in enumerate(tqdm(sigma_train_dataloader, 0)):
            input_2D, gt_3D, output_3D,distance_sigma = data
            [input_2D, gt_3D, output_3D,distance_sigma]=get_varialbe('train', [input_2D, gt_3D, output_3D,distance_sigma])
            
            sigma_pred=avg_model(input_2D)

            sigma_gt=distance_sigma
            # sigma_gt=torch.clamp(distance_sigma,max=3)
                
            loss = L2(sigma_pred, sigma_gt)
            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() , N)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        logging.info('epoch: %d, lr: %.7f, loss: %.4f' % (epoch, lr, loss_all['loss'].avg))
        print('e: %d, lr: %.7f, loss: %.4f' % (epoch, lr, loss_all['loss'].avg))

        torch.save(avg_model.state_dict(),f'{opt.save_ckpt_path}/avg_model_{opt.shhpe_model}.pth')

        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay

def load_dict(model,ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} is not exist")
        exit()
    model_dict = model.state_dict()
    print(f"INFO: load pretrained model_dict {ckpt_path}")
    pre_dict = torch.load(ckpt_path)

    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    manualSeed = opt.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print("lr: ", opt.lr)
    print("batch_size: ", opt.batch_size)
    print("GPU: ", opt.gpu)


    # load dataset
    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(opt.save_ckpt_path, 'train.log'), level=logging.INFO)

    if opt.dataset == 'h36m':
        dataset_path = opt.root_path + 'data_3d_' + opt.dataset + '.npz'
        dataset = Human36mDataset(dataset_path, opt)
        actions = define_actions(opt.actions)

        if opt.train:
            train_data = Fusion(opt, dataset, opt.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                            shuffle=True, num_workers=int(opt.workers), pin_memory=True)
        test_data = Fusion(opt, dataset, opt.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                            shuffle=False, num_workers=int(opt.workers), pin_memory=True)
        
    elif opt.dataset == '3dhp':
        dataset_path = opt.root_path + 'data_3d_' + opt.dataset + '.npz'
        dataset = Mpi_inf_3dhp_Dataset(dataset_path, opt)
        actions = define_actions_3dhp(opt.actions, 0)

        if opt.train:
            train_data = Fusion_3dhp(opt, dataset, opt.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                            shuffle=True, num_workers=int(opt.workers), pin_memory=True)
        test_data = Fusion_3dhp(opt, dataset, opt.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                            shuffle=False, num_workers=int(opt.workers), pin_memory=True)
        
        
    # load sh_hpe model
    adj = adj_mx_from_skeleton(dataset.skeleton())

    if opt.shhpe_model=='htnet':
        sh_model = HTNet(adj).cuda()
        
    elif opt.shhpe_model=='mgcn_lw':
        sh_model =MGCN_lw(adj, hid_dim=128, num_layers=4,nodes_group=None).cuda()
    else:
        print(f"Error:{opt.shhpe_model} is not exist")
        exit()

    model_params = 0
    for parameter in sh_model.parameters():
        model_params += parameter.numel()
    print('INFO: Single hypothesis model Trainable parameter count (M):', model_params / 1000000)

    sh_model_path=f"ckpt/shhpe_model_{opt.shhpe_model}.pth"
    load_dict(sh_model,sh_model_path)

    # load avg_model
    if opt.shhpe_model=='htnet':
        avg_model=ModulatedGCN(adj,hid_dim=256).cuda()
        
    elif opt.shhpe_model=='mgcn_lw':
        avg_model=LinearModel(num_stage=1).cuda()
    else:
        print(f"Error:{opt.shhpe_model} is not exist")
        exit()
    model_params = 0
    for parameter in avg_model.parameters():
        model_params += parameter.numel()
    print('INFO: Avg_model Trainable parameter count (M):', model_params / 1000000)

    if not opt.train:
        print(f'======>>>>> start to test in {opt.dataset} <<<<<======')
        avg_model_path = f"ckpt/avg_model_{opt.shhpe_model}.pth"
        load_dict(avg_model,avg_model_path)

        with torch.no_grad():
            p1, p2, pck = test(opt,actions, test_dataloader, sh_model,avg_model)
        
        # mpi_inf_3dhp test
        if opt.dataset == '3dhp':
            
            print('MPI-INF-3DHP [  pck: %.2f  ]' % (pck))
        else:
            print('Human3.6M  [  p1: %.2f, p2: %.2f ]' % (p1,p2))
    
    else:
        if opt.dataset == 'h36m':
            print('======>>>>> start to train <<<<<======')
        else:
            print(f"Error: MPI-INF-3DHP is only used for testing")
            exit()
        all_param = []
        lr = opt.lr
        all_param += list(avg_model.parameters())
        optimizer = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.317, patience=5, verbose=True)

        sigma_train_dataloader=get_sigma_dataloader(opt,sh_model,train_dataloader,opt.shhpe_model,type='train')
        # sigma_test_dataloader=get_sigma_dataloader(opt,sh_model,test_dataloader,type='test')

        train(opt, actions, avg_model, lr, optimizer, sigma_train_dataloader)







