import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value

def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]

def define_actions_3dhp( action, train ):
  if train:
    actions = ["Seq1", "Seq2"]
  else:
    actions = ["Seq1"]

    return actions

# def define_error_list(actions):
#     error_sum = {}
#     error_sum.update({actions[i]:
#         {'p1':AccumLoss(), 'p2':AccumLoss()}
#         for i in range(len(actions))})
#     return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: 
        {'p1':AccumLoss(), 'p2':AccumLoss(), 'pck':AccumLoss()} 
        for i in range(len(actions))})
    return error_sum

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var

def print_error(data_type, action_error_sum):
    mean_error_p1, mean_error_p2, pck = 0, 0, 0
    mean_error_p1, mean_error_p2, pck = print_error_action(action_error_sum, data_type)

    return mean_error_p1, mean_error_p2, pck

def print_error_action(action_error_sum, data_type):
    mean_error_each = {'p1': 0.0, 'p2': 0.0, 'pck': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss(), 'pck': AccumLoss()}

    if data_type.startswith('3dhp'):
        print("{0:=^12} {1:=^10} {2:=^8} {3:=^8}".format("Action", "p#1 mm", "p#2 mm", "PCK"))
    else:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():

        print("{0:<12} ".format(action), end="")
            
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        mean_error_each['pck'] = action_error_sum[action]['pck'].avg * 100.0
        mean_error_all['pck'].update(mean_error_each['pck'], 1)

        if data_type.startswith('3dhp'):
            print("{0:>6.2f} {1:>10.2f} {2:>10.2f} ".format(
                mean_error_each['p1'], mean_error_each['p2'], 
                mean_error_each['pck']))
        else:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if data_type.startswith('3dhp'):
        print("{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} ".format("Average", 
            mean_error_all['p1'].avg, mean_error_all['p2'].avg,
            mean_error_all['pck'].avg))
    else:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
            mean_error_all['p2'].avg))

    if data_type.startswith('3dhp'):
        return mean_error_all['p1'].avg, mean_error_all['p2'].avg,  \
                mean_error_all['pck'].avg
    else:
        return mean_error_all['p1'].avg, mean_error_all['p2'].avg, 0


def save_model_refine(previous_name, save_dir,epoch, data_threshold, model, model_name):#
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)

    return previous_name


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    return previous_name



def save_model_epoch(previous_name, save_dir, epoch, data_threshold, model):
    # if os.path.exists(previous_name):
    #     os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    return previous_name






