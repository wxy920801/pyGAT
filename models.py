import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nhid1, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.mid_att = GraphAttentionLayer(nhid * nheads, nhid1*nheads, dropout=dropout, alpha=alpha    , concat=False)

        self.out_att = GraphAttentionLayer(nhid1 * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        
    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.mid_att(x, adj)
        x = self.out_att(x, adj)
        return x
