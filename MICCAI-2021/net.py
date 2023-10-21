from torch import nn

from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified, Normalize
from mean_shift import GBMS_RNN


class MSNet(nn.Module):
    def __init__(self, num_classes):
        super(MSNet, self).__init__()
        self.layers = nn.Sequential(

            # SPDTransform(58, 48, 1),
            # SPDRectified(),
            # SPDTransform(48, 36, 1),
            # SPDRectified(),
            # SPDTransform(36, 24, 1),
            # SPDRectified(),
            # SPDTransform(24, 16, 1),
            # SPDTangentSpace(24, vectorize_all=False),
            # Normalize(),

            # whole brain
            # SPDTransform(268, 128, 1),
            # SPDRectified(),
            # SPDTransform(128, 64, 1),
            # SPDRectified(),
            # SPDTransform(64, 32, 1),
            # SPDRectified(),
            # SPDTransform(32, 16, 1),
            # Normalize(),

            SPDTransform(10, 9, 1),
            SPDRectified(),
            SPDTransform(9, 8, 1),
            SPDRectified(),
            SPDTransform(8, 6, 1),
            SPDTangentSpace(),
            Normalize(),

            GBMS_RNN(normalize=True),
            GBMS_RNN(normalize=True),
            GBMS_RNN(normalize=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
