from rdkit import Chem
from data_process.data_process import smile_to_graph
demo_list = []
drugs = []
drug_smiles = []
for d in demo_list:
    lg = Chem.MolToSmiles(Chem.MolFromSmiles(d), isomericSmiles=True)
    drugs.append(lg)
    drug_smiles.append(d)
print("drugs",len(drugs))
smile_graph = {}
for smile in drug_smiles:
    g = smile_to_graph(smile)
    if(smile_graph.get(smile) is not None):
        print(smile)
    smile_graph[smile] = g
print('effective durgs',len(smile_graph))