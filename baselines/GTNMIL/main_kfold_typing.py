#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from itertools import chain
from random import sample, shuffle

from utils.dataset import GraphDataset
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options

# from utils.saliency_maps import *

from models.GraphTransformer import Classifier
from models.weight_init import weight_init

from sklearn.metrics import roc_auc_score, f1_score
from statistics import stdev

args = Options().parse()
n_class = args.n_class

# torch.cuda.synchronize()
torch.backends.cudnn.deterministic = True

data_path = args.data_path
model_path = args.model_path
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path): os.mkdir(log_path)
task_name = args.task_name

print(task_name)
###################################
train = args.train
test = args.test
graphcam = args.graphcam
print("train:", train, "test:", test, "graphcam:", graphcam)

##### Load datasets
print("preparing datasets and dataloaders......")
batch_size = args.batch_size

all_normal_set = args.all_normal_set
all_tumor_set = args.all_tumor_set
ids_set_normal = open(all_normal_set).readlines()
ids_set_tumor = open(all_tumor_set).readlines()

kfold_ids_set_normal = np.array_split(np.array(ids_set_normal), args.num_fold)
kfold_ids_set_tumor = np.array_split(np.array(ids_set_tumor), args.num_fold)

acc_fold_list, auc_fold_list, f1_fold_list = [], [], []

for i_fold in range(0, args.num_fold):

    ids_train = []
    ids_val = []
    ids_test = []

    for idx_fold in range(0, args.num_fold):
        if idx_fold != i_fold:
            ids_train.append(kfold_ids_set_normal[idx_fold])
            ids_train.append(kfold_ids_set_tumor[idx_fold])
        else:
            temp_normal = np.array_split(kfold_ids_set_normal[i_fold],2)
            temp_tumor = np.array_split(kfold_ids_set_tumor[i_fold],2)
            ids_val.append(temp_normal[0])
            ids_test.append(temp_normal[1])
            ids_val.append(temp_tumor[0])
            ids_test.append(temp_tumor[1])
    
    ids_train = np.array(list(chain.from_iterable(ids_train)))
    ids_val = np.array(list(chain.from_iterable(ids_val)))
    ids_test = np.array(list(chain.from_iterable(ids_test)))
    shuffle(ids_train)
    shuffle(ids_val)
    shuffle(ids_test)

    if train:
        dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train)
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=4, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
        total_train_num = len(dataloader_train) * batch_size

    dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=4, collate_fn=collate, shuffle=False, pin_memory=True)
    total_val_num = len(dataloader_val) * batch_size

    dataset_test = GraphDataset(os.path.join(data_path, ""), ids_test)
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, num_workers=4, collate_fn=collate, shuffle=False, pin_memory=True)
    total_test_num = len(dataloader_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ############ creating models #############
    print("creating models......")

    num_epochs = args.num_epochs
    learning_rate = args.lr
    print(num_epochs)

    model = Classifier(n_class)
    model = nn.DataParallel(model)
    if args.resume:
        print('load model{}'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))

    if torch.cuda.is_available():
        model = model.cuda()
    #model.apply(weight_init)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)       # best:5e-4, 4e-3
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,100], gamma=0.1) # gamma=0.3  # 30,90,130 # 20,90,130 -> 150
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0.000005)
    ##################################

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()


    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

    trainer = Trainer(n_class)
    evaluator = Evaluator(n_class)

    best_pred = 0.0
    best_test = 0.0
    best_f1 = 0.0
    best_auc = 0.0

    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.
        total = 0.

        current_lr = optimizer.param_groups[0]['lr']
        print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch+1, current_lr, best_pred))

        if train:
            
            for i_batch, sample_batched in enumerate(dataloader_train):
                optimizer.zero_grad()
                #scheduler(optimizer, i_batch, epoch, best_pred)

                preds,labels,loss, out = trainer.train(sample_batched, model)
                del sample_batched

                if not torch.isnan(loss):
                    loss.sum().backward()
                    optimizer.step()
                scheduler.step(epoch)

                train_loss += float(loss.sum().item())
                total += int(len(labels.detach().cpu()))

                trainer.metrics.update(labels.detach().cpu(), preds.detach().cpu())
                #trainer.plot_cm()

                

                if (i_batch + 1) % args.log_interval_local == 0:
                    print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total, total_train_num, train_loss / total, trainer.get_scores()))
                    # trainer.plot_cm()
                
                del loss, preds, labels

        print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total_train_num, total_train_num, train_loss / total, trainer.get_scores()))
        # trainer.plot_cm()


        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                
                print("evaluating...")

                total = 0.
                batch_idx = 0

                for i_batch, sample_batched in enumerate(dataloader_val):
                    #pred, label, _ = evaluator.eval_test(sample_batched, model)
                    preds, labels, _, out = evaluator.eval_test(sample_batched, model, graphcam)
                    del sample_batched
                    total += int(len(labels.detach().cpu()))

                    evaluator.metrics.update(labels.detach().cpu(), preds.detach().cpu())

                    if (i_batch + 1) % args.log_interval_local == 0:
                        print('[%d/%d] val agg acc: %.3f' % (total, total_val_num, evaluator.get_scores()))
                        # evaluator.plot_cm()
                    
                    del preds, labels, _

                print('[%d/%d] val agg acc: %.3f' % (total_val_num, total_val_num, evaluator.get_scores()))
                # evaluator.plot_cm()

                torch.cuda.empty_cache()

                val_acc = evaluator.get_scores()
                if val_acc > best_pred: 
                    best_pred = val_acc
                    if not test:
                        print("saving model...")
                        torch.save(model.state_dict(), model_path + task_name + ".pth")

                log = ""
                log = log + 'epoch [{}/{}] ------ acc: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, trainer.get_scores(), evaluator.get_scores()) + "\n"

                log += "================================\n"
                print(log)

                f_log.write(log)
                f_log.flush()

                writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()}, epoch+1)

        # trainer.reset_metrics()
        # evaluator.reset_metrics()
            model.eval()
            with torch.no_grad():

                print("testing...")

                total = 0.
                batch_idx = 0

                test_labels = []
                test_predictions = []

                for i_batch, sample_batched in enumerate(dataloader_test):
                    preds, labels, _, out = evaluator.eval_test(sample_batched, model, graphcam)
                    del sample_batched
                    total += int(len(labels))

                    evaluator.metrics.update(labels, preds)

                    test_labels.extend(labels.detach().cpu().numpy())
                    test_predictions.extend(preds.detach().cpu().numpy())

                    del preds, labels, _

                test_labels = np.array(test_labels)
                test_predictions = np.array(test_predictions)

                auc_value = roc_auc_score(test_labels, test_predictions)
                f1 = f1_score(y_pred=test_predictions, y_true=test_labels, average='binary')

                
                print('[%d/%d] Test agg acc: %.3f' % (total_test_num, total_test_num, evaluator.get_scores()))
                evaluator.plot_cm()

                test_acc = evaluator.get_scores()
                if test_acc > best_test: 
                    best_test = test_acc

                if auc_value > best_auc:
                    best_auc = auc_value

                if f1 > best_f1:
                    best_f1 = f1

                log = ""
                log = log + 'epoch [{}/{}] ------ acc: Test = {:.4f}'.format(epoch+1, num_epochs, evaluator.get_scores()) + "\n"

                log += "================================\n"
                print(log)
    

    acc_fold_list.append(best_test)
    auc_fold_list.append(best_auc)
    f1_fold_list.append(best_f1)
    
    evaluator.reset_metrics()

print('all acc test:')
for r in acc_fold_list:
    print(r)
print('mean acc test: ', np.mean(np.array(acc_fold_list)))
print('stdev acc test: ', stdev(acc_fold_list))
print(" ")

print('all f1 test:')
for r in f1_fold_list:
    print(r)
print('mean f1 test', np.mean(np.array(f1_fold_list)))
print('stdev f1 test: ', stdev(f1_fold_list))
print(" ")

print('all auc test:')
for r in auc_fold_list:
    print(r)
print('mean auc test', np.mean(np.array(auc_fold_list)))
print('stdev auc test: ', stdev(auc_fold_list))
print(" ")