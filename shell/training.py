import sys
import torch.nn as nn
from models import HeteroDTA as GNNNet
from tools.utils import *
from tools.emetrics import *
from data_process.data_process import create_dataset_for_5folds

USE_CUDA = torch.cuda.is_available()
datasets = [['davis', 'kiba'][int(sys.argv[1])]]
cuda_name = ['cuda:0', 'cuda:1'][int(sys.argv[2])]
dataset_type = ['original', 'cold_drug', 'cold_protein', 'cold_pair'][int(sys.argv[3])]
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
NUM_EPOCHS = 2000
print('datasets:', datasets)
print('cuda_name:', cuda_name)
print('dataset_type:', dataset_type)
print('BATCH_SIZE: ', TRAIN_BATCH_SIZE)
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('use_cuda', USE_CUDA)

models_dir = 'weights'
results_dir = '../results'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = GNNNet()
model.to(device)
model_st = GNNNet.__name__

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for dataset in datasets:
    train_data, test_data = create_dataset_for_5folds(dataset, dataset_type, model_st)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collate)

    best_mse = 10
    best_test_mse = 10
    best_epoch = -1
    if int(sys.argv[3]) == 0:
        model_file_name = 'weights/' + dataset + '.model'
    else:
        model_file_name = 'weights/' + dataset + '.model'
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1, loss_fn, TRAIN_BATCH_SIZE)
        print('predicting for test data')
        G, P = predicting(model, device, test_loader)
        ret = get_mse(G, P)
        if ret < best_mse:
            best_mse = ret
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print('mse improved at epoch ', best_epoch, '; best_test_mse:', best_mse, dataset)
        else:
            print('No improvement since epoch ', best_epoch, '; best_test_mse:', best_mse, dataset)