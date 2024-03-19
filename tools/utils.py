import os
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import numpy as np
import json
from tqdm import tqdm
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None, target_sequence=None,
                 motif_graph=None, model_st=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_graph, target_sequence, motif_graph, model_st)
    @property
    def raw_file_names(self):
        pass
    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']
    def download(self):
        pass
    def _download(self):
        pass
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    def process(self, xd, target_key, y, smile_graph, target_graph, target_sequence, clique_graph, model_st):
        assert (len(xd) == len(target_key) and len(xd) == len(y))
        data_list_mol = []
        data_list_motif = []
        data_list_pro = []
        data_len = len(xd)
        pbar = tqdm(total=data_len)
        for i in range(data_len):
            smiles = xd[i]
            tar_key = target_key[i]
            tar_seq = target_sequence[i]
            labels = y[i]
            c_size, features, GEM, edge_index, edge_attr = smile_graph[smiles]
            clique_size, clique_features, clique_edge_index = clique_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight, target_esm_graph_embedding = \
                target_graph[tar_key]
            GCNData_mol = DATA.Data(x=torch.FloatTensor(np.array(features)),
                                    GEM=torch.FloatTensor(GEM),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    edge_attr=torch.FloatTensor(edge_attr),
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData_motif = DATA.Data(x=torch.Tensor(np.array(clique_features)),
                                      edge_index=torch.LongTensor(clique_edge_index).transpose(1, 0),
                                      y=torch.FloatTensor([labels]))
            GCNData_motif.__setitem__('motif_size', torch.LongTensor([clique_size]))
            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(target_edge_weight),
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
            GCNData_pro.graph_embedding = torch.FloatTensor([target_esm_graph_embedding])
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)
            data_list_motif.append(GCNData_motif)
            pbar.update(1)
        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
            data_list_motif = [data for data in data_list_motif if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
            data_list_motif = [data for data in data_list_motif if self.pre_filter(data)]

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        self.data_motif = data_list_motif

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx], self.data_motif[idx]
def train(model, device, train_loader, optimizer, epoch, loss_fn, train_batch_size=512):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    log_interval = 10
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        data_motif = data[2].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro, data_motif)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * train_batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_motif = data[2].to(device)
            output = model(data_mol, data_pro, data_motif)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, batchC