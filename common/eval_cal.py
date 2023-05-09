import torch
import numpy as np


def mpjpe(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=-1))


def pck(predicted, target):
    assert predicted.shape == target.shape
    threshold = 150.0 / 1000

    frame_num = predicted.shape[1]*1.0
    joints_num = predicted.shape[-2]*1.0
    
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)

    t = torch.Tensor([threshold]).cuda()
    out = (dis < t).float() * 1
    pck = out.sum() / joints_num / frame_num

    return pck


def auc(predicted, target):
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)
    outall = 0
    threshold = 150

    frame_num = predicted.shape[1]*1.0
    joints_num = predicted.shape[-2]*1.0

    for i in range(threshold):
        t = torch.Tensor([float(i)/1000]).cuda()
        out = (dis < t).float() * 1
        outall+=out.sum() /joints_num / frame_num

    outall = outall/threshold
    
    return outall


def test_calculation_muyilhyp(predicted, target, action, error_sum, data_type, subject):
    if data_type == 'h36m':
        error_sum = mpjpe_by_action_p1_mutilhpy(predicted, target, action, error_sum)
        error_sum = mpjpe_by_action_p2_mutilhyp(predicted, target, action, error_sum)
    elif data_type.startswith('3dhp'):
        # error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
        # error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

        error_sum = mpjpe_by_action_pck_mutilhyp(predicted, target, action, error_sum)

    return error_sum

def mpjpe_by_action_p1_mutilhpy(predicted, target, action, action_error_sum):

    num = predicted.size(0)
    dist = torch.min(torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2),dim=1)[0]
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
    return action_error_sum


def mpjpe_by_action_p2_mutilhyp(predicted, target, action, action_error_sum):

    num,hyp,j,c = predicted.shape
    target=target.repeat(1,hyp,1,1)
    pred=predicted.contiguous().view(-1,17,3).detach().cpu().numpy()
    gt=target.contiguous().view(-1,17,3).detach().cpu().numpy()

    dist=p_mpjpe(pred,gt).reshape(num, hyp).min(axis=1)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        # action_error_sum[action_name][2] += num*np.mean(dist)
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            # action_error_sum[action_name][2] += dist[i].item()
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
    return action_error_sum


def mpjpe_by_action_pck(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = pck(predicted, target)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['pck'].update(dist.item(), num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['pck'].update(dist[i].item(), 1)
            
    return action_error_sum

def pck_mutilhyp(predicted, target):
    assert predicted.shape == target.shape
    threshold = 150.0 / 1000
    
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)

    t = torch.Tensor([threshold]).cuda()
    out = (dis < t).float() * 1
    pck = torch.mean(out.view(predicted.shape[0],-1),dim=1)

    return pck

def mpjpe_by_action_pck_mutilhyp(predicted, target, action, action_error_sum):
    num,h,j,c = predicted.shape

    target=target.repeat(1,h,1,1).contiguous().view(-1,1,17,3)
    predicted=predicted.contiguous().view(-1,1,17,3)

    assert predicted.shape == target.shape

    dist = pck_mutilhyp(predicted, target)

    dist=torch.sum(torch.max(dist.view(num,h),dim=1)[0])

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['pck'].update(dist.item(), num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['pck'].update(dist[i].item(), 1)
            
    return action_error_sum

def mpjpe_by_action_auc(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = auc(predicted, target)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['auc'].update(dist.item(), num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['auc'].update(dist[i].item(), 1)
            
    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY 
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)




