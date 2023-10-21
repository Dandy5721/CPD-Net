import torch
from torch import nn
from spdnet.spd import Normalize


class GBMS_RNN(nn.Module):
    def __init__(self, bandwidth=0.1, normalize=True):
        super(GBMS_RNN, self).__init__()
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth))
        self.normalize = None
        if normalize:
            self.normalize = Normalize()

    def forward(self, X):
        bandwidth = self.bandwidth
        if self.normalize:
            W = torch.exp((X @ X.transpose(-2, -1) - 1) / (bandwidth * bandwidth))
        else:
            pair_dis = torch.cdist(X, X)
            pair_dis_square = pair_dis**2
            W = torch.exp(-0.5 * pair_dis_square / (bandwidth * bandwidth))
        D = W.sum(dim=-1).diag_embed()
        D_inv = D.inverse()
        X = (X.transpose(-2, -1) @ W @ D_inv).transpose(-2, -1)
        if self.normalize:
            X = self.normalize(X)
        output = X
        return output


def cosine_similarity(input):
    output = input @ input.transpose(-2, -1) * 0.5 + 0.5
    return output


def similarity_loss(input, targets, alpha=0):
    similarity = cosine_similarity(input)
    identity_matrix = targets.unsqueeze(-2) == targets.unsqueeze(-2).transpose(-2, -1)
    loss = (1 - similarity) * identity_matrix + torch.clamp(
        similarity - alpha, min=0
    ) * (~identity_matrix)
    loss = torch.mean(loss)
    return loss
