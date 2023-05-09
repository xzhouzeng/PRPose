
from einops.einops import rearrange
import torch
import torch.nn as nn
from models.htnet.Block import Hiremixer


class HTNet(nn.Module):
    def __init__(self, adj):
        super().__init__()

        layers, channel, d_hid, length  = 3, 240, 1024, 1
        self.num_joints_in, self.num_joints_out = 17, 17

        self.patch_embed = nn.Linear(2, channel)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, channel))
        self.Hiremixer = Hiremixer(adj, layers, channel, d_hid, length=length)
        self.fcn = nn.Linear(channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.Hiremixer(x) # (bs,17,240)

        x = self.fcn(x) # (bs,17,3)
        x = x.view(x.shape[0], -1, self.num_joints_out, x.shape[2]) # (bs,1,17,3)

        return x


