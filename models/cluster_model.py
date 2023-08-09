import torch
import torch.nn as nn
from models.GraphConvSkip import GraphConvSkip
import torch.nn.functional as F


class cluster_model(nn.Module):
    def __init__(self, num_feats, num_hiddens, num_communites):
        super(cluster_model, self).__init__()
        self.name = "cluster_model"
        self.num_feats = num_feats
        self.num_hiddens = num_hiddens
        self.num_communites = num_communites
        
        self.softmax = nn.Softmax(dim=1)
        self.GraphConvSkip = GraphConvSkip(self.num_feats, self.num_hiddens, torch.nn.SELU())
        self.linear = torch.nn.Linear(self.num_hiddens, self.num_communites)
        

    def forward(self, norm_adj, feat):
        S = self.softmax(self.linear(self.GraphConvSkip(norm_adj, feat)))
        return S