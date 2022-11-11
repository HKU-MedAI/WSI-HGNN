from statistics import stdev
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, f1_score
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from itertools import chain

from math import sqrt

def get_bag_feats_v2(wsi, args):
    # if isinstance(wsi, str):
        # if feats is a path, load it
    feats = wsi.iloc[0]
    feats = np.load(feats)
    
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = wsi.iloc[1]
    else:
        if int(wsi.iloc[1])<=(len(label)-1):
            label[int(wsi.iloc[1])] = 1
        
    return label, feats

def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats

def train(train_df, milnet, criterion, optimizer, args):
    milnet.train()
    csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    bc = 0
    Tensor = torch.cuda.FloatTensor
    for i in range(len(train_df)):
        optimizer.zero_grad()
        if args.dataset == 'COAD':
            label, feats = get_bag_feats_v2(train_df.iloc[i], args)

        elif args.dataset == 'BRCA':
            label, feats = get_bag_feats_v2(train_df.iloc[i], args)

        else:
            label, feats = get_bag_feats(train_df.iloc[i], args)
        feats = dropout_patches(feats, args.dropout_patch)
        bag_label = Variable(Tensor(np.array(label)))
        bag_feats = Variable(Tensor(np.array(feats)))
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)

        if args.model == 'dsmil':
            max_prediction, _ = torch.max(ins_prediction, 0)        
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
        elif args.model == 'abmil':
            loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))

        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def test(test_df, milnet, criterion, optimizer, args):
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            if args.dataset == 'COAD':
                label, feats = get_bag_feats_v2(test_df.iloc[i], args)
            elif args.dataset == 'BRCA':
                label, feats = get_bag_feats_v2(test_df.iloc[i], args)
            else:
                label, feats = get_bag_feats(test_df.iloc[i], args)
            bag_label = Variable(Tensor(np.array(label)))
            bag_feats = Variable(Tensor(np.array(feats)))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            if args.model == 'dsmil':
                max_prediction, _ = torch.max(ins_prediction, 0)        
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
                loss = 0.5*bag_loss + 0.5*max_loss
            elif args.model == 'abmil':
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])

            if args.model == 'dsmil':
                test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            elif args.model == 'abmil':
                test_predictions.extend([(torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)

    f1 = f1_score(y_pred=test_predictions, y_true=test_labels, average='macro')
    c_auc = roc_auc_score(y_score=test_predictions, y_true=test_labels, multi_class="ovr", average='macro')
    
    # return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, f1
    return total_loss / len(test_df), avg_score, c_auc, thresholds_optimal, f1

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def stddev(lst):
    mean = float(sum(lst)) / len(lst)
    return sqrt(sum((x - mean)**2 for x in lst) / len(lst))

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_fold', default=5, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(1,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import model.abmil as mil

    if args.model == 'dsmil':
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        state_dict_weights = torch.load('init.pth')
        milnet.load_state_dict(state_dict_weights, strict=False)
    
    elif args.model == 'abmil':
        milnet = mil.BClassifier_(args.feats_size, args.num_classes).cuda()
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    if args.dataset == 'COAD':
        normal_path = pd.read_csv('datasets/COAD/list/normal_list_base.csv').iloc[0:,:]
        tumor_path = pd.read_csv('datasets/COAD/list/tumor_list_base.csv').iloc[0:,:]

        stage_0_path = pd.read_csv('datasets/COAD/list/stage_0_list_base.csv').iloc[0:,:]
        stage_1_path = pd.read_csv('datasets/COAD/list/stage_1_list_base.csv').iloc[0:,:]
        stage_2_path = pd.read_csv('datasets/COAD/list/stage_2_list_base.csv').iloc[0:,:]
        stage_3_path = pd.read_csv('datasets/COAD/list/stage_3_list_base.csv').iloc[0:,:]
    
    if args.dataset == 'BRCA':
        normal_path = pd.read_csv('datasets/BRCA/list/normal_list_base.csv').iloc[0:,:]
        tumor_path = pd.read_csv('datasets/BRCA/list/tumor_list_base.csv').iloc[0:,:]

        stage_0_path = pd.read_csv('datasets/BRCA/list/stage_0_list_base.csv').iloc[0:,:]
        stage_1_path = pd.read_csv('datasets/BRCA/list/stage_1_list_base.csv').iloc[0:,:]
        stage_2_path = pd.read_csv('datasets/BRCA/list/stage_2_list_base.csv').iloc[0:,:]
        stage_3_path = pd.read_csv('datasets/BRCA/list/stage_3_list_base.csv').iloc[0:,:]

    kfold_stage_0_path = np.array_split(np.array(stage_0_path), args.num_fold)
    kfold_stage_1_path = np.array_split(np.array(stage_1_path), args.num_fold)
    kfold_stage_2_path = np.array_split(np.array(stage_2_path), args.num_fold)
    kfold_stage_3_path = np.array_split(np.array(stage_3_path), args.num_fold)


    best_score = 0
    save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))

    acc_fold_list, auc_fold_list, f1_fold_list = [], [], []

    for i_fold in range(0, args.num_fold):
        train_path, val_path, test_path = [], [], []
        for idx in range(0, args.num_fold):
            if idx != i_fold:
                train_path.append(kfold_stage_0_path[idx].tolist())
                train_path.append(kfold_stage_1_path[idx].tolist())
                train_path.append(kfold_stage_2_path[idx].tolist())
                train_path.append(kfold_stage_3_path[idx].tolist())

            else:
                stage_0_path = np.array_split(kfold_stage_0_path[idx], 2)
                stage_1_path = np.array_split(kfold_stage_1_path[idx], 2)
                stage_2_path = np.array_split(kfold_stage_2_path[idx], 2)
                stage_3_path = np.array_split(kfold_stage_3_path[idx], 2)
                val_path.append(stage_0_path[0].tolist())
                val_path.append(stage_1_path[0].tolist())
                val_path.append(stage_2_path[0].tolist())
                val_path.append(stage_3_path[0].tolist())

                test_path.append(stage_0_path[1].tolist())
                test_path.append(stage_1_path[1].tolist())
                test_path.append(stage_2_path[1].tolist())
                test_path.append(stage_3_path[1].tolist())

        train_path = pd.DataFrame(chain.from_iterable(train_path))
        val_path = pd.DataFrame(chain.from_iterable(val_path))
        test_path = pd.DataFrame(chain.from_iterable(test_path))

        for epoch in range(1, args.num_epochs):
            train_path = shuffle(train_path).reset_index(drop=True)
            val_path = shuffle(val_path).reset_index(drop=True)
            test_path = shuffle(test_path).reset_index(drop=True)
            train_loss_bag = train(train_path, milnet, criterion, optimizer, args) # iterate all bags
            val_loss_bag, val_avg_score, val_aucs, val_thresholds_optimal, val_f1_score = test(val_path, milnet, criterion, optimizer, args)

            print('\r Epoch [%d/%d] train loss: %.4f ' % (epoch, args.num_epochs, train_loss_bag))
            
            # print('\r val loss: %.4f, val average score: %.4f, val f1 score: %.4f, val AUC: ' % 
            #     (val_loss_bag, val_avg_score, val_f1_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(val_aucs))) 

            print('\r val loss: %.4f, val average score: %.4f, val f1 score: %.4f, val AUC: %.4f' % 
                (val_loss_bag, val_avg_score, val_f1_score, val_aucs)) 

            scheduler.step()
            current_score = (val_aucs + val_avg_score)/2
            if current_score >= best_score:
                best_score = current_score
                save_name = os.path.join(save_path, str(run+1)+'.pth')
                torch.save(milnet.state_dict(), save_name)
                if args.dataset=='TCGA-lung':
                    print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
                else:
                    print('Best model saved at: ' + save_name)
                    print('Best val thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(val_thresholds_optimal)))
        
        test_loss_bag, avg_score, aucs, thresholds_optimal, test_f1_score = test(test_path, milnet, criterion, optimizer, args)
        # print('\r test loss: %.4f, test average score: %.4f, test f1 score: %.4f,test AUC: ' % 
        #             (test_loss_bag, avg_score, test_f1_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        print('\r test loss: %.4f, test average score: %.4f, test f1 score: %.4f,test AUC: %.4f' % 
                    (test_loss_bag, avg_score, test_f1_score, aucs)) 
        
        acc_fold_list.append(avg_score)
        auc_fold_list.append(aucs)
        f1_fold_list.append(test_f1_score)


    print('mean acc fold value')
    print(acc_fold_list)
    print('acc: ' + str(sum(acc_fold_list)/ len(acc_fold_list)))
    print('std: ' + str(stdev(acc_fold_list)))

    print('mean auc fold value')
    print(auc_fold_list)
    print('acc: ' + str(sum(auc_fold_list)/ len(auc_fold_list)))
    print('std: ' + str(stdev(auc_fold_list)))

    print('mean f1 fold value')
    print(f1_fold_list)
    print('acc: ' + str(sum(f1_fold_list)/ len(f1_fold_list)))
    print('std: ' + str(stdev(f1_fold_list)))

if __name__ == '__main__':
    main()