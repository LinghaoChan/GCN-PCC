import os
from data import ModelNet40
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sklearn.metrics as metrics
from net import GCNPCNET



parameters = {
    'dataset' : 'modelnet40_ply_hdf5_2048', 
    'opt' : 'SGD', 
    'data' : 'data', 
    'cuda' : True, 
    'batch_size' : 32, 
    'test_batch_size' : 16, 
    'epochs' : 100, 
    'lr' : 0.001,
    'momentum' : 0.9,
    'dropout' : 0.5, 
    'emb_dims' : 1024, 
    'num_points' : 1024, 
    'k' : 20
}

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def train(parameters):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=parameters['num_points']), num_workers=8, batch_size=parameters['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=parameters['num_points']), num_workers=8, batch_size=parameters['batch_size'], shuffle=True, drop_last=False)
    device = torch.device("cuda" if parameters['cuda'] else "cpu")

    model = GCNPCNET(parameters).to(device)
    print(str(model))
    
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if parameters['opt'] == 'SGD':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=parameters['lr']*100, momentum=parameters['momentum'], weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=1e-4)


    scheduler = CosineAnnealingLR(opt, parameters['epochs'], eta_min=parameters['lr'])
    
    criterion = cal_loss

    best_test_acc = 0

    for epoch in range(parameters['epochs']):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            # print(data.shape)
            # print(label.shape)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        # io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        # io.cprint(outstr)


    
train(parameters)   