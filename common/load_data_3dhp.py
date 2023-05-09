import sys
import torch.utils.data as data

from common.camera import *
from common.utils import deterministic_random
from common.generator import ChunkedGenerator


class Fusion_3dhp(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):

        self.opt = opt
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(
            ',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d = \
                self.fetch(dataset, self.train_list, subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size, self.cameras_train, self.poses_train,
                                              self.poses_train_2d,
                                              self.stride, pad=self.pad,
                                              augment=opt.data_augmentation,
                                              reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            print('Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d = \
                self.fetch(dataset, self.test_list, subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size, self.cameras_test, self.poses_test,
                                              self.poses_test_2d, self.stride,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            self.key_index = self.generator.saved_index
            print('Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list):
        for subject in folder_list:
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                for i in range(len(anim['positions'])):
                    pos_3d = anim['positions'][i]

                    pos_3d[:, 1:] -= pos_3d[:, :1]

                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        keypoints_pth = self.root_path + 'data_2d_' + \
            self.data_type + '_'+self.keypoints_name+'.npz'

        keypoints = np.load(keypoints_pth, allow_pickle=True)
        keypoints = keypoints['positions_2d'].item()

        self.kps_left, self.kps_right = [
            4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
        self.joints_left, self.joints_right = [
            4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

        for subject in folder_list:
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(
                subject)
            for action in dataset[subject].keys():
                assert action in keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                         subject)
                for cam_idx in range(len(keypoints[subject][action])):
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]

                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, item in enumerate(keypoints[subject][action]):
                    kps = keypoints[subject][action][cam_idx]

                    if subject == 'TS5' or subject == 'TS6':
                        res_w = 1920
                        res_h = 1080
                    else:
                        res_w = 2048
                        res_h = 2048

                    if self.crop_uv == 0:
                        kps[..., :2] = normalize_screen_coordinates(
                            kps[..., :2], w=res_w, h=res_h)

                    keypoints[subject][action][cam_idx] = kps

        return keypoints

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}

        out_camera_params = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]

                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(
                        poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(
                    round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(
                    0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start +
                                                      n_frames:stride]

                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start +
                                                          n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]

                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(
            seq_name, start_3d, end_3d, flip, reverse)

        if self.train == False and self.test_aug:
            _, _, input_2D_aug, _, _, _ = self.generator.get_batch(
                seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            input_2D = np.concatenate(
                (np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)

        return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, cam_ind
