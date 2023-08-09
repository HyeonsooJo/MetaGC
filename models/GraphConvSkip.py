import torch
from torch import nn
from torch.nn import init


class GraphConvSkip(nn.Module):
    def __init__(self, in_feats, out_feats, act):
        super(GraphConvSkip, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
    
        self.weight_1 = nn.Parameter(torch.Tensor(self._in_feats, self._out_feats))
        self.weight_2 = nn.Parameter(torch.Tensor(self._in_feats, self._out_feats))
        self.bias = nn.Parameter(torch.Tensor(self._out_feats))
        
        self.reset_parameters()
        self.act = act
        
            
    def reset_parameters(self):
        if self.weight_1 is not None:
            init.xavier_uniform_(self.weight_1)
        if self.weight_2 is not None:
            init.xavier_uniform_(self.weight_2)
        if self.bias is not None:
            init.zeros_(self.bias)
    
    def forward(self, norm_adj, feat):
        return self.act(norm_adj @ feat @ self.weight_1 + feat @ self.weight_2 + self.bias)
