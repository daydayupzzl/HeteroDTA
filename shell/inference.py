import sys
import matplotlib.pyplot as plt
from tools.emetrics import get_cindex, get_rm2, get_ci, get_mse, get_rmse, get_pearson, get_spearman
from tools.utils import *
from models import HeteroDTA as GNNNet
from data_process.data_process import create_dataset_for_test
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
            # data = data.to(device)
            output = model(data_mol, data_pro, data_motif)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
def load_model(model_path):
    model = torch.load(model_path)
    return model
def calculate_metrics(Y, P, result_file_name, dataset='davis'):
    cindex = get_cindex(Y, P)
    cindex2 = get_ci(Y, P)
    rm2 = get_rm2(Y, P)
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)
    result_str = ''
    result_str += dataset + '\r\n'
    result_str += 'rmse:' + str(rmse) + ' ' + ' mse:' + str(mse) + ' ' + ' pearson:' + str(
        pearson) + ' ' + 'spearman:' + str(spearman) + ' ' + 'ci:' + str(cindex) + ' ' + 'rm2:' + str(rm2)
    print(result_str)
    open(result_file_name, 'w').writelines(result_str)

if __name__ == '__main__':
    dataset = ['davis', 'kiba'][int(sys.argv[1])]
    cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][int(sys.argv[2])]
    dataset_type = ['original', 'cold_drug', 'cold_protein', 'cold_pair'][int(sys.argv[3])]
    model_st = GNNNet.__name__
    TEST_BATCH_SIZE = 512
    models_dir = 'weights'
    results_dir = '../results'
    print('dataset:', dataset)
    print('cuda_name:', cuda_name)
    print('dataset_type:', dataset_type)

    test_data = create_dataset_for_test(dataset, dataset_type, model_st)
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    if int(sys.argv[3]) == 0:
        model_file_name = 'weights/' + dataset + '.model'
        result_file_name = 'results/' + dataset + '.txt'
    else:
        model_file_name = 'weights/' + dataset + '.model'
        result_file_name = 'results/' + dataset + '.txt'
    model = GNNNet()
    model.to(device)
    model.load_state_dict(torch.load(model_file_name, map_location=cuda_name), False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collate)

    Y, P = predicting(model, device, test_loader)
    calculate_metrics(Y, P, result_file_name, dataset)
