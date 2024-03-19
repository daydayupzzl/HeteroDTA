import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool
class Feature_Fusion(nn.Module):
    def __init__(self, channels=128 * 2, r=4):
        super(Feature_Fusion, self).__init__()
        inter_channels = int(channels // r)
        layers = [nn.Linear(channels, inter_channels), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(inter_channels, channels)]
        self.att1 = nn.Sequential(*layers)
        self.att2 = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fd, fp):
        w1 = self.sigmoid(self.att1(fd + fp))
        fout1 = fd * w1 + fp * (1 - w1)
        w2 = self.sigmoid(self.att2(fout1))
        fout2 = fd * w2 + fp * (1 - w2)
        return fout2

class GNNNet_DTA(nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=32, num_features_motif=92,
                 num_features_esm=1280, output_dim=128, dropout=0.2, hidden_dim=1024):
        super().__init__()
        self.mol_convs = nn.ModuleList(
            [GATConv(num_features_mol * 2 ** i, num_features_mol * 2 ** (i + 1), 6) for i in range(3)])
        self.mol_fcs = nn.ModuleList(
            [nn.Linear(num_features_mol * 2 ** 3, hidden_dim), nn.Linear(hidden_dim, output_dim)])

        self.motif_convs = nn.ModuleList(
            [GATConv(num_features_motif * 2 ** i, num_features_motif * 2 ** (i + 1)) for i in range(3)])
        self.motif_fcs = nn.ModuleList(
            [nn.Linear(num_features_motif * 2 ** 3, hidden_dim), nn.Linear(hidden_dim, output_dim)])

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_convs = nn.ModuleList(
            [GATConv(num_features_pro * 2 ** i, num_features_pro * 2 ** (i + 1)) for i in range(2)])
        self.pro_fcs = nn.ModuleList(
            [nn.Linear(num_features_pro * 2 ** 3, hidden_dim), nn.Linear(hidden_dim, output_dim)])

        self.esm_fcs = nn.ModuleList([nn.Linear(num_features_esm, 512), nn.Linear(512, output_dim)])

        self.dtf = Feature_Fusion(channels=output_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 2, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, n_output)
        )

    def forward(self, data_mol, data_pro, data_motif):
        mol_x, mol_edge_index, mol_batch, edge_attr = data_mol.GEM, data_mol.edge_index, data_mol.batch, data_mol.edge_attr
        motif_x, motif_edge_index, motif_batch = data_motif.x, data_motif.edge_index, data_motif.batch
        pro_x, pro_edge_index, pro_edge_weight, pro_emb, pro_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.graph_embedding, data_pro.batch

        x = self.mol_convs[0](mol_x, mol_edge_index, edge_attr)
        for conv in self.mol_convs[1:]:
            x = F.relu(conv(x, mol_edge_index, edge_attr))
        x = global_max_pool(x, mol_batch)
        for fc in self.mol_fcs:
            x = fc(F.relu(x))
            x = F.dropout(x, 0.2, self.training)

        m_x = self.motif_convs[0](motif_x, motif_edge_index)
        for conv in self.motif_convs[1:]:
            m_x = F.relu(conv(m_x, motif_edge_index))
        m_x = global_max_pool(m_x, motif_batch)
        for fc in self.motif_fcs:
            m_x = fc(F.relu(m_x))
            m_x = F.dropout(m_x, 0.2, self.training)

        xt = self.pro_conv1(pro_x, pro_edge_index, pro_edge_weight)
        for conv in self.pro_convs:
            xt = F.relu(conv(xt, pro_edge_index))
        xt = global_max_pool(xt, pro_batch)
        for fc in self.pro_fcs:
            xt = fc(F.relu(xt))
            xt = F.dropout(xt, 0.2, self.training)

        esm = self.esm_fcs[0](F.relu(pro_emb))
        esm = F.dropout(esm, 0.2, self.training)
        esm = self.esm_fcs[1](esm)

        ligand_x = torch.cat((x, m_x), 1)
        protein_x = torch.cat((xt, esm), 1)
        xc = self.dtf(ligand_x, protein_x)
        xc = self.classifier(xc)
        return xc