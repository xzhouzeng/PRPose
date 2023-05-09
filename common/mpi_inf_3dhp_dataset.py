import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates

mpi_inf_3dhp_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

class Mpi_inf_3dhp_Dataset(MocapDataset):
    def __init__(self, path, opt,remove_static_joints=True):
        super().__init__(fps=50, skeleton=mpi_inf_3dhp_skeleton)
        self.train_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
        self.test_list = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

        data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                }
        if remove_static_joints:
            joints_to_remove=[4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]

            kept_joints = self._skeleton.remove_joints(joints_to_remove)
            # for subject in self._data.keys():
            #     for action in self._data[subject].keys():
            #         s = self._data[subject][action]
            #         s['positions'] = s['positions'][:, kept_joints]

            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8


