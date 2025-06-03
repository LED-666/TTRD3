import torch.nn as nn
from codes.models.TransferNet.MFAM import MFAM
from codes.models.TransferNet.STTG import STTG


class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        self.MFAM = MFAM(dim_in=3, n_feats=64)
        self.STTG = STTG()

    def forward(self, lrsr=None, ref=None, refsr=None):
        _, _, lrsr_lv3  = self.MFAM((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.MFAM((refsr.detach() + 1.) / 2.)
        ref_lv1, ref_lv2, ref_lv3 = self.MFAM((ref.detach() + 1.) / 2.)

        T = self.STTG(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        return T