import torch.nn as nn
import torch


class meta_model(nn.Module):
    def __init__(self, num_feat, num_hiddens):
        super(meta_model, self).__init__()
        self.name = "meta_model"

        self.linear_link_1 = nn.Linear(num_feat, num_hiddens)
        self.linear_link_2 = nn.Linear(num_hiddens, num_hiddens)

        self.linear_unlink_1 = nn.Linear(num_feat, num_hiddens)
        self.linear_unlink_2 = nn.Linear(num_hiddens, num_hiddens)

        self.linear_aa_1 = nn.Linear(num_feat, num_hiddens)
        self.linear_aa_2 = nn.Linear(num_hiddens, num_hiddens)

        self.linear_mod_1 = nn.Linear(num_feat, num_hiddens)
        self.linear_mod_2 = nn.Linear(num_hiddens, num_hiddens)

        self.weighted_sum = nn.Parameter(torch.Tensor(3))
        nn.init.zeros_(self.weighted_sum)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def forward(self, feat, adj, aa, mod):
        latent_link = self.relu(self.linear_link_1(feat))
        latent_link = self.linear_link_2(latent_link)

        latent_aa = self.relu(self.linear_aa_1(feat))
        latent_aa = self.linear_aa_2(latent_aa)

        latent_mod = self.relu(self.linear_mod_1(feat))
        latent_mod = self.linear_mod_2(latent_mod)


        link_context = self.sigmoid(latent_link @ latent_link.T)
        aa_context = self.sigmoid((latent_aa @ latent_aa.T) * aa)
        mod_context = self.sigmoid((latent_mod @ latent_mod.T) * mod)

        link_context = torch.unsqueeze(link_context, dim=2)
        aa_context = torch.unsqueeze(aa_context, dim=2)
        mod_context = torch.unsqueeze(mod_context, dim=2)

        weight_link = torch.cat([link_context, aa_context, mod_context], dim=2)
        weighted_sum = self.softmax(self.weighted_sum)
        weight_link = weight_link @ weighted_sum

        latent_unlink = self.relu(self.linear_unlink_1(feat))
        latent_unlink = self.linear_unlink_2(latent_unlink)
        weight_unlink = self.sigmoid(latent_unlink @ latent_unlink.T)
        return (weight_link - weight_unlink) * adj + weight_unlink