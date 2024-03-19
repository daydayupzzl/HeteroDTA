import pandas as pd
import json, pickle
from collections import OrderedDict
import networkx as nx
from tqdm import tqdm
from tools.utils import *
import random
import sys
import openpyxl

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
def encoding_unk(x, allowable_set):
    list = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            list[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        list[-1] = True
    return list
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def seq_feature(seq):
    residue_feature = []
    for residue in seq:
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)
    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]
    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature
def atom_features(atom):
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
class GraphConvConstants(object):
    possible_atom_list = [
        'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
        'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
    ]
    possible_numH_list = [0, 1, 2, 3, 4]
    possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
    possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
    possible_hybridization_list = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
    possible_number_radical_e_list = [0, 1, 2]
    possible_chirality_list = ['R', 'S']
    bond_fdim_base = 6
import numpy as np

def bond_features_deep_chemm(bond, use_chirality=False, use_extended_chirality=False):
    bond_feats = [
        bond.GetBondTypeAsDouble() == 1.0, bond.GetBondTypeAsDouble() == 2.0,
        bond.GetBondTypeAsDouble() == 3.0, bond.GetIsAromatic(), bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats += one_hot_encoding_unk(str(bond.GetStereo()), GraphConvConstants.possible_bond_stereo)
    if use_extended_chirality:
        bond_feats += one_hot_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return np.array(bond_feats, dtype=np.float32)
def smile_to_graph(index, smile, dataset):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = [atom_features(atom) / sum(atom_features(atom)) for atom in mol.GetAtoms()]
    with open(f"data/{dataset}/{dataset}_gem_emb.txt", "r") as f:
        data = json.load(f)
    GEM = np.array(eval(list(data.values())[index]))
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    g = nx.Graph(edges)
    mol_adj = nx.to_numpy_matrix(g, nodelist=range(c_size))
    np.fill_diagonal(mol_adj, 1)
    edge_index = np.column_stack(np.where(mol_adj >= 0.5))
    edges_attr = []
    for i, j in edge_index:
        bond = mol.GetBondBetweenAtoms(int(i), int(j))
        if int(i) == int(j):
            edges_attr.append([0] * 6)
        elif bond is not None:
            edges_attr.append(bond_features_deep_chemm(bond))
    return c_size, features, GEM, edge_index, edges_attr
def smile_to_motifGraph(smile):
    mol = Chem.MolFromSmiles(smile)
    clique, edge = cluster_graph(mol)
    c_features = [clique_features(cq, edge, idx, smile) / sum(clique_features(cq, edge, idx, smile)) for idx, cq in enumerate(clique)]
    clique_size = len(clique)
    return clique_size, c_features, edge
def cluster_graph(mol):
    n_atoms = mol.GetNumAtoms()
    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)
    nei_list = [[] for _ in range(n_atoms)]
    edges = []
    for i in range(len(cliques) - 1):
        for j in range(i + 1, len(cliques)):
            if len(set(cliques[i]) & set(cliques[j])) != 0:
                edges.append([i, j])
                edges.append([j, i])
    return cliques, edges
def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')

def clique_features(clique, edges, clique_idx, smile):
    num_atoms = len(clique)
    num_edges = sum(clique_idx in edge for edge in edges)
    mol = Chem.MolFromSmiles(smile)
    atoms = []
    num_hydrogens = 0
    num_implicit_valence = 0
    for idx in clique:
        atom = mol.GetAtomWithIdx(idx)
        atoms.append(atom.GetSymbol())
        num_hydrogens += atom.GetTotalNumHs()
        num_implicit_valence += atom.GetImplicitValence()
    is_ring = int(len(clique) > 2)
    is_bond = int(len(clique) == 2)
    atom_encoding = encoding_unk(atoms, ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                         'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                         'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                         'Pt', 'Hg', 'Pb', 'X'])
    num_atoms_encoding = one_hot_encoding_unk(num_atoms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    num_edges_encoding = one_hot_encoding_unk(num_edges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    num_hydrogens_encoding = one_hot_encoding_unk(num_hydrogens, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    num_implicit_valence_encoding = one_hot_encoding_unk(num_implicit_valence, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    return np.array(atom_encoding + num_atoms_encoding + num_edges_encoding +
                    num_hydrogens_encoding + num_implicit_valence_encoding +
                    [is_ring] + [is_bond])
import os
import numpy as np

def sequence_to_graph(target_key, target_sequence, distance_dir, esm_embeds=None):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    distance_map = np.load(contact_map_file)
    np.fill_diagonal(distance_map, 1)
    np.fill_diagonal(distance_map[:, 1:], 1)
    index_row, index_col = np.where(distance_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
        target_edge_distance.append(distance_map[i, j])
    target_feature = seq_feature(target_sequence)
    esm_embedding = np.mean(esm_embeds[target_key], axis=0)
    return target_size, target_feature, np.array(target_edge_index), np.array(target_edge_distance), esm_embedding
def valid_target(key, dataset):
    contact_dir = 'pre_process/' + dataset + '/distance_map'
    contact_file = os.path.join(contact_dir, key + '.npy')
    if os.path.exists(contact_file):
        return True
    else:
        return False
def create_dataset_for_5folds(dataset, version_experiment, model_st):
    if model_st is None:
        print('Invalid model_st. Program terminated.')
        sys.exit()
    print('dataset:', dataset, 'create_dataset_for_nofolds')
    process_dir = os.path.join('', 'pre_process')
    pro_distance_dir = os.path.join(process_dir, dataset, 'distance_map')
    dataset_path = 'data/' + dataset + '/'
    if version_experiment == 'original':
        train_fold_origin = json.load(open(dataset_path + '/train_fold_setting1.txt'))
        test_fold = json.load(open(dataset_path + '/test_fold_setting1.txt'))
    elif version_experiment == 'cold_drug':
        train_fold_origin = json.load(open(dataset_path + '/train_cold_drug_setting.txt'))
        test_fold = json.load(open(dataset_path + '/test_cold_drug_setting.txt'))
    elif version_experiment == 'cold_protein':
        train_fold_origin = json.load(open(dataset_path + '/train_cold_protein_setting.txt'))
        test_fold = json.load(open(dataset_path + '/test_cold_protein_setting.txt'))
    elif version_experiment == 'cold_pair':
        train_fold_origin = json.load(open(dataset_path + '/train_cold_pair_setting.txt'))
        test_fold = json.load(open(dataset_path + '/test_cold_pair_setting.txt'))
    else:
        print('Invalid version_experiment. Program terminated.')
        sys.exit()
    train_folds = []
    print('len of train_fold_setting', len(train_fold_origin))
    for i in range(len(train_fold_origin)):
        train_folds = train_folds + train_fold_origin[i]
    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)

    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    opts = ['train', 'test']
    valid_train_count = 0
    valid_test_count = 0
    for opt in opts:
        if opt == 'train':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[train_folds], cols[train_folds]
            train_fold_entries = []
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                train_fold_entries.append(ls)
                valid_train_count += 1
            if version_experiment == 'original':
                csv_file = 'data/' + dataset + '_Nofold_' + opt + '.csv'
            elif version_experiment == 'cold_drug':
                csv_file = 'data/' + dataset + '_Nofold_cold_drug_' + opt + '.csv'
            elif version_experiment == 'cold_protein':
                csv_file = 'data/' + dataset + '_Nofold_cold_protein_' + opt + '.csv'
            elif version_experiment == 'cold_pair':
                csv_file = 'data/' + dataset + '_Nofold_cold_pair_' + opt + '.csv'
            data_to_csv(csv_file, train_fold_entries)
        elif opt == 'test':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[test_fold], cols[test_fold]
            temp_test_entries = []
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                temp_test_entries.append(ls)
                valid_test_count += 1
            if version_experiment == 'original':
                csv_file = 'data/' + dataset + '_test.csv'
            elif version_experiment == 'cold_drug':
                csv_file = 'data/' + dataset + '_cold_drug_test.csv'
            elif version_experiment == 'cold_protein':
                csv_file = 'data/' + dataset + '_cold_protein_test.csv'
            elif version_experiment == 'cold_pair':
                csv_file = 'data/' + dataset + '_cold_pair_test.csv'
            data_to_csv(csv_file, temp_test_entries)
    compound_iso_smiles = drugs
    target_key = prot_keys

    smile_graph = {}
    with tqdm(total=len(compound_iso_smiles), desc="processing drug_graph") as pbar:
        for index, smile in enumerate(compound_iso_smiles):
            g = smile_to_graph(index, smile, dataset)
            smile_graph[smile] = g
            pbar.update(1)

    motif_graph = {}
    with tqdm(total=len(compound_iso_smiles), desc="processing motif_graph") as pbar:
        for smile in compound_iso_smiles:
            g = smile_to_motifGraph(smile)
            motif_graph[smile] = g
            pbar.update(1)

    target_graph = {}
    input_file = 'data/' + dataset + '/esm_node_embeds_' + dataset + '.npy'
    esm_embeds = np.load(input_file, allow_pickle=True).item()
    with tqdm(total=len(target_key), desc="processing protein_graph") as pbar:
        for key in target_key:
            if not valid_target(key, dataset):
                continue
            g = sequence_to_graph(key, proteins[key], pro_distance_dir, esm_embeds)
            target_graph[key] = g
            pbar.update(1)
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')
    if version_experiment == 'original':
        df_train_fold = pd.read_csv('data/' + dataset + '_Nofold_' + 'train' + '.csv')
    elif version_experiment == 'cold_drug':
        df_train_fold = pd.read_csv('data/' + dataset + '_Nofold_cold_drug_' + 'train' + '.csv')
    elif version_experiment == 'cold_protein':
        df_train_fold = pd.read_csv('data/' + dataset + '_Nofold_cold_protein_' + 'train' + '.csv')
    elif version_experiment == 'cold_pair':
        df_train_fold = pd.read_csv('data/' + dataset + '_Nofold_cold_pair_' + 'train' + '.csv')

    train_drugs, train_prot_keys, train_prot_sequence, train_Y = list(df_train_fold['compound_iso_smiles']), list(
        df_train_fold['target_key']), list(df_train_fold['target_sequence']), list(df_train_fold['affinity'])
    train_drugs, train_prot_keys, train_prot_sequence, train_Y = np.asarray(train_drugs), np.asarray(
        train_prot_keys), np.asarray(train_prot_sequence), np.asarray(train_Y)
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y, smile_graph=smile_graph, target_graph=target_graph, motif_graph=motif_graph,
                               target_sequence=train_prot_sequence, model_st=model_st)

    if version_experiment == 'original':
        df_test_fold = pd.read_csv('data/' + dataset + '_test.csv')
    elif version_experiment == 'cold_drug':
        df_test_fold = pd.read_csv('data/' + dataset + '_cold_drug_test.csv')
    elif version_experiment == 'cold_protein':
        df_test_fold = pd.read_csv('data/' + dataset + '_cold_protein_test.csv')
    elif version_experiment == 'cold_pair':
        df_test_fold = pd.read_csv('data/' + dataset + '_cold_pair_test.csv')

    test_drugs, test_prots_keys, test_prot_sequence, test_Y = list(df_test_fold['compound_iso_smiles']), list(
        df_test_fold['target_key']), list(df_test_fold['target_sequence']), list(df_test_fold['affinity'])
    test_drugs, test_prots_keys, test_prot_sequence, test_Y = np.asarray(test_drugs), np.asarray(
        test_prots_keys), np.asarray(test_prot_sequence), np.asarray(
        test_Y)
    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=test_drugs,
                              target_key=test_prots_keys, y=test_Y, smile_graph=smile_graph,
                              target_graph=target_graph, motif_graph=motif_graph, target_sequence=test_prot_sequence,
                              model_st=model_st)
    return train_dataset, test_dataset
def create_dataset_for_test(dataset, version_experiment, model_st):
    if model_st is None:
        print('Invalid model_st. Program terminated.')
        sys.exit()
    dataset_path = 'data/' + dataset + '/'
    if version_experiment == 'original':
        test_fold = json.load(open(dataset_path + '/test_fold_setting1.txt'))
    elif version_experiment == 'cold_drug':
        test_fold = json.load(open(dataset_path + '/test_cold_drug_setting.txt'))
    elif version_experiment == 'cold_protein':
        test_fold = json.load(open(dataset_path + '/test_cold_protein_setting.txt'))
    elif version_experiment == 'cold_pair':
        test_fold = json.load(open(dataset_path + '/test_cold_pair_setting.txt'))
    else:
        print('Invalid version_experiment. Program terminated.')
        sys.exit()

    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')

    process_dir = os.path.join('', 'pre_process')
    pro_distance_dir = os.path.join(process_dir, dataset, 'distance_map')

    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    valid_test_count = 0
    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[test_fold], cols[test_fold]
    temp_test_entries = []
    for pair_ind in range(len(rows)):
        if not valid_target(prot_keys[cols[pair_ind]], dataset):
            continue
        ls = []
        ls += [drugs[rows[pair_ind]]]
        ls += [prots[cols[pair_ind]]]
        ls += [prot_keys[cols[pair_ind]]]
        ls += [affinity[rows[pair_ind], cols[pair_ind]]]
        temp_test_entries.append(ls)
        valid_test_count += 1
    if version_experiment == 'original':
        csv_file = 'data/' + dataset + '_test.csv'
    elif version_experiment == 'cold_drug':
        csv_file = 'data/' + dataset + '_cold_drug_test.csv'
    elif version_experiment == 'cold_protein':
        csv_file = 'data/' + dataset + '_cold_protein_test.csv'
    elif version_experiment == 'cold_pair':
        csv_file = 'data/' + dataset + '_cold_pair_test.csv'
    data_to_csv(csv_file, temp_test_entries)
    print('dataset:', dataset)
    print('test entries:', len(test_fold), 'effective test entries', valid_test_count)

    compound_iso_smiles = drugs
    target_key = prot_keys

    smile_graph = {}
    with tqdm(total=len(compound_iso_smiles), desc="Processing drug graph") as pbar:
        for index, smile in enumerate(compound_iso_smiles):
            g = smile_to_graph(index, smile, dataset)
            smile_graph[smile] = g
            pbar.update(1)
    motif_graph = {}
    with tqdm(total=len(compound_iso_smiles), desc="Processing motif graph") as pbar:
        for smile in compound_iso_smiles:
            g = smile_to_motifGraph(smile)
            motif_graph[smile] = g
            pbar.update(1)
    target_graph = {}
    input_file = 'data/' + dataset + '/esm_node_embeds_' + dataset + '.npy'
    esm_embeds = np.load(input_file, allow_pickle=True).item()
    with tqdm(total=len(target_key), desc="Processing protein") as pbar:
        for key in target_key:
            if not valid_target(key, dataset):
                continue
            g = sequence_to_graph(key, proteins[key], pro_distance_dir, esm_embeds)
            target_graph[key] = g
            pbar.update(1)

    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    if version_experiment == 'original':
        df_test = pd.read_csv('data/' + dataset + '_test.csv')
    elif version_experiment == 'cold_drug':
        df_test = pd.read_csv('data/' + dataset + '_cold_drug_test.csv')
    elif version_experiment == 'cold_protein':
        df_test = pd.read_csv('data/' + dataset + '_cold_protein_test.csv')
    elif version_experiment == 'cold_pair':
        df_test = pd.read_csv('data/' + dataset + '_cold_pair_test.csv')

    test_drugs, test_prot_keys, test_prot_sequence, test_Y = list(df_test['compound_iso_smiles']), list(df_test['target_key']),list(df_test['target_sequence']),  list(
        df_test['affinity'])
    test_drugs, test_prot_keys, test_prot_sequence, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_prot_sequence), np.asarray(test_Y)
    test_dataset = DTADataset(root='data', dataset=dataset + '_test', xd=test_drugs, y=test_Y,
                              target_key=test_prot_keys, smile_graph=smile_graph, target_graph=target_graph,
                              motif_graph=motif_graph, target_sequence=test_prot_sequence,model_st=model_st)
    return test_dataset