import os
import os.path
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import time as sys_time
import copy
import argparse
import joblib
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from statistics import stdev
from itertools import chain
from random import shuffle

from RAConv import RAConv
from IHPool import IHPool


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_classes, no_of_class, drop_out_ratio=0.2, pool1_ratio=0.2, pool2_ratio=4,
                 pool3_ratio=3, mpool_method="global_mean_pool"):
        super(GCN, self).__init__()
        self.conv1 = RAConv(in_channels=in_feats, out_channels=out_classes)
        self.conv2 = RAConv(in_channels=out_classes, out_channels=out_classes)

        self.pool_1 = IHPool(in_channels=out_classes, ratio=pool1_ratio, select='inter', dis='ou')
        self.pool_2 = IHPool(in_channels=out_classes, ratio=pool2_ratio, select='inter', dis='ou')

        if mpool_method == "global_mean_pool":
            self.mpool = global_mean_pool
        elif mpool_method == "global_max_pool":
            self.mpool = global_max_pool
        elif mpool_method == "global_att_pool":
            att_net = nn.Sequential(nn.Linear(out_classes, out_classes // 2), nn.ReLU(), nn.Linear(out_classes // 2, 1))
            self.mpool = GlobalAttention(att_net)

        self.lin1 = torch.nn.Linear(out_classes, out_classes // 2)
        self.lin2 = torch.nn.Linear(out_classes // 2, no_of_class) # number of classes

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out_ratio)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = LayerNorm(in_feats)

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index, node_type, data_id, tree, x_y_index = data.edge_index_tree_8nb, data.node_type, data.data_id, data.node_tree, data.x_y_index
        x_y_index = x_y_index * 2 - 1

        x = self.norm(x)
        x = self.conv1(x, edge_index, node_type)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x, edge_index_1, edge_weight, batch, cluster_1, node_type_1, tree_1, score_1, x_y_index_1 = self.pool_1(x=x,
                                                                                                                edge_index=edge_index,
                                                                                                                node_type=node_type,
                                                                                                                tree=tree,
                                                                                                                x_y_index=x_y_index)
        batch = edge_index_1.new_zeros(x.size(0))
        x1 = self.mpool(x, batch)
        x = self.conv2(x, edge_index_1, node_type_1)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)

        x, edge_index_2, edge_weight, batch, cluster_2, node_type_2, tree_2, score_2, x_y_index_2 = self.pool_2(x,
                                                                                                                edge_index_1,
                                                                                                                node_type=node_type_1,
                                                                                                                tree=tree_1,
                                                                                                                x_y_index=x_y_index_1)
        batch = edge_index_2.new_zeros(x.size(0))
        x2 = self.mpool(x, batch)

        x = x1 + x2

        x = self.lin1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.softmax(x)

        return x, (cluster_1, cluster_2), (node_type_1, node_type_2), (score_1, score_2), (tree_1, tree_2), (x_y_index_1, x_y_index_2), (edge_index_1, edge_index_2)


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def return_train_val_index(train_index, label_list, seed):
    label_list_temp = []
    for index in train_index:
        label_list_temp.append(label_list[index])
    train_index_split, val_index_spilt, _, _ = train_test_split(train_index, label_list_temp, test_size=0.25,
                                                                random_state=seed, stratify=label_list_temp)
    return train_index_split, val_index_spilt


def reuturn_data_label(all_data, data_index, patiens_list, label_list, batch_size):
    data_array = []
    label_array = []
    for item in data_index:
        temp_patient_name = patiens_list[item]
        temp_data = all_data[temp_patient_name]
        label_array.append(label_list[item])
        data_array.append(temp_data)
    step = batch_size
    data_array_temp = [data_array[i:i + step] for i in range(0, len(data_array), step)]
    label_array_temp = [label_array[i:i + step] for i in range(0, len(label_array), step)]
    return data_array_temp, label_array_temp


def return_acc(prediction_array, label_array):
    prediction_array = np.array(prediction_array)
    label_array = np.array(label_array)
    a = []
    for i in label_array:
        a.append(np.where(i == 1)[0][0])
    correct_num = (prediction_array == a).sum()
    len_array = len(prediction_array)
    return correct_num / len_array

def return_f1(prediction_array, label_array):
    prediction_array = np.array(prediction_array)
    label_array = np.array(label_array)
    a = []
    for i in label_array:
        a.append(np.where(i == 1)[0][0])
    score = f1_score(y_pred=prediction_array, y_true=a, average='macro')
    return score


def return_auc(possibility_array, label_array):
    label_array = np.array(label_array)
    possibility_array = possibility_array.argmax(axis=1)
    label_array = label_array.argmax(axis=1)
    print(possibility_array)
    print(label_array)
    fpr, tpr, thresholds = roc_curve(label_array, possibility_array)
    from sklearn.metrics import auc
    aucroc = auc(fpr, tpr)
    # return roc_auc_score(y_true=label_array, y_score=possibility_array, average='macro', multi_class='ovr')
    return aucroc

def val_test_block(model, loss_fun, device, data_for_val_test, label_for_val_test):
    model.eval()
    with torch.no_grad():
        label_array_for_val_test = []
        prediction_array_for_val_test = []
        possibility_array_for_val_test = []
        total_loss_for_val_test = 0
        for index_batch_for_val_test, data_batch_for_val_test in enumerate(data_for_val_test):
            label_batch_for_val_test = label_for_val_test[index_batch_for_val_test]
            for index_temp, data_val_test_single in enumerate(data_batch_for_val_test):
                label_val_test_single = label_batch_for_val_test[index_temp]
                label_array_for_val_test.append(label_val_test_single)
                data_val_test_single = data_val_test_single.to(device)
                label_val_test_single = torch.tensor([label_val_test_single]).to(device)
                output_for_val_test, _1, _2, _3, _4, _5, _6 = model(data_val_test_single)
                _, prediction_for_val_test = torch.max(output_for_val_test, 1)
                loss_for_val_test = loss_fun(output_for_val_test, label_val_test_single)
                total_loss_for_val_test += loss_for_val_test
                prediction_array_for_val_test.append(prediction_for_val_test.cpu().item())
                # possibility_array_for_val_test.append(output_for_val_test[0][1].detach().cpu().item())
                possibility_array_for_val_test.append(output_for_val_test.detach().cpu().numpy())

        possibility_array_for_val_test = np.squeeze(np.array(possibility_array_for_val_test),1)
        acc_for_val_test = return_acc(prediction_array_for_val_test, label_array_for_val_test)
        auc_for_val_test = return_auc(possibility_array_for_val_test, label_array_for_val_test)
        f1_score_for_val_test = return_f1(prediction_array_for_val_test, label_array_for_val_test)
        return label_array_for_val_test, prediction_array_for_val_test, possibility_array_for_val_test, total_loss_for_val_test, auc_for_val_test, acc_for_val_test, f1_score_for_val_test


class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(sys_time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


def main(args):
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)

    all_data_path = args.all_data_path
    patient_and_label_path = args.patient_and_label_path
    repeat_num = args.repeat_num
    divide_seed = args.divide_seed
    drop_out_ratio = args.drop_out_ratio
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    out_classes = args.out_classes
    no_of_class = args.number_of_classes
    pool1_ratio = args.pool1_ratio
    pool2_ratio = args.pool2_ratio
    pool3_ratio = args.pool3_ratio
    mpool_method = args.mpool_method

    loss_fun = torch.nn.CrossEntropyLoss()
    all_data = joblib.load(all_data_path)
    patient_and_label = joblib.load(patient_and_label_path)
    patiens_list = []
    label_list = []

    for item in patient_and_label:
        patiens_list.append(item)
        label_arr = np.zeros(no_of_class)
        label_arr[patient_and_label[item]] = 1

        label_list.append(label_arr)

    ######################
    # split patient and label list to train, val, and test

    train_list = [os.path.basename(x).split('.')[0] for x in open('data/BRCA_kimia_lv0/list/homogeneous_train.txt', 'r').readlines()]
    val_list = [os.path.basename(x).split('.')[0] for x in open('data/BRCA_kimia_lv0/list/homogeneous_val.txt', 'r').readlines()]
    test_list = [os.path.basename(x).split('.')[0] for x in open('data/BRCA_kimia_lv0/list/homogeneous_test.txt', 'r').readlines()]

    all_list = train_list + val_list + test_list
    ids_set_normal = []
    ids_set_tumor = []

    for i_all_list in all_list:
        if i_all_list in patient_and_label:
            if patient_and_label[i_all_list] == 0:
                ids_set_normal.append(i_all_list)
            else:
                ids_set_tumor.append(i_all_list)

    kfold_ids_set_normal = np.array_split(np.array(ids_set_normal), repeat_num)
    kfold_ids_set_tumor = np.array_split(np.array(ids_set_tumor), repeat_num)

    all_fold_auc = []
    all_fold_acc = []
    all_fold_f1 = []
    all_pre_result = []

    for repeat_num_temp in range(repeat_num):

        ######################
        # split patient and label list to train, val, and test
        ids_train = []
        ids_val = []
        ids_test = []

        train_index = []
        val_index = []
        test_index = []

        for idx_fold in range(0, repeat_num):
            if idx_fold != repeat_num_temp:
                ids_train.append(kfold_ids_set_normal[idx_fold])
                ids_train.append(kfold_ids_set_tumor[idx_fold])
            else:
                temp_normal = np.array_split(kfold_ids_set_normal[repeat_num_temp],2)
                temp_tumor = np.array_split(kfold_ids_set_tumor[repeat_num_temp],2)
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

        for i in ids_train:
            train_index.append(patiens_list.index(i))

        for j in ids_val:
            val_index.append(patiens_list.index(j))

        for k in ids_test:
            test_index.append(patiens_list.index(k))
        #############################################


        pre_result = {}

        seed = divide_seed + repeat_num_temp
        setup_seed(seed)
     
        best_auc_val_fold = 0
 
        data_for_train, label_for_train = reuturn_data_label(all_data, train_index, patiens_list, label_list,
                                                                batch_size)
        data_for_val, label_for_val = reuturn_data_label(all_data, val_index, patiens_list, label_list,
                                                            batch_size)
        data_for_test, label_for_test = reuturn_data_label(all_data, list(test_index), patiens_list, label_list,
                                                            batch_size)
        print("train_index_split: " + str(len(train_index)) + " val_index_split: " + str(len(val_index))
                + " test_index_split: " + str(len(test_index)))


        model = GCN(in_feats=1024, n_hidden=256, out_classes=out_classes, no_of_class=no_of_class ,drop_out_ratio=drop_out_ratio,
                    pool1_ratio=pool1_ratio, pool2_ratio=pool2_ratio, pool3_ratio=pool3_ratio,
                    mpool_method=mpool_method)

        model = model.to(device)
        
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        for epoch_num in range(epochs):
            model.train()
            prediction_array = []
            label_array = []
            possibility_array = []
            total_loss_for_train = 0
            for index_batch_for_train, data_batch_for_train in enumerate(data_for_train):
                batch_loss = 0
                label_batch_for_train = label_for_train[index_batch_for_train]
                for index1, data_train_single in enumerate(data_batch_for_train):
                    label_train_single = label_batch_for_train[index1]
                    label_array.append(label_train_single)
                    data_train_single = data_train_single.to(device)
                    label_train_single = torch.tensor([label_train_single]).to(device)
                    output_for_train, _1, _2, _3, _4, _5, _6 = model(data_train_single)
                    _, prediction_for_train = torch.max(output_for_train, 1)
                    loss = loss_fun(output_for_train, label_train_single)
                    batch_loss += loss
                    total_loss_for_train += loss
                    prediction_array.append(prediction_for_train.cpu().item())
                    possibility_array.append(output_for_train.detach().cpu().numpy())
        
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            possibility_array = np.squeeze(np.array(possibility_array),1)

            acc = return_acc(prediction_array, label_array)
            auc = return_auc(possibility_array, label_array)
            # auc = return_auc(prediction_array, label_array)

            print("Train acc: " + str(acc))

            label_array_for_val, prediction_array_for_val, possibility_array_for_val, total_loss_for_val, auc_for_val, acc_for_val, f1_for_val = val_test_block(
                model, loss_fun, device, data_for_val, label_for_val)


            if auc_for_val >= best_auc_val_fold and (auc > 0.8 or epoch_num > (epochs - 10)):
                best_auc_val_fold = auc_for_val
                print(best_auc_val_fold)
                t_model = copy.deepcopy(model)

            print(
                "epoch：{:2d}，train_loss：{:.4f}, train_auc：{:.4f}, val_loss：{:.4f}, val_auc：{:.4f}, val_acc：{:.4f}, val_f1：{:.4f}".format(
                    epoch_num, total_loss_for_train / len(train_index), auc,
                    total_loss_for_val / len(val_index), auc_for_val, acc_for_val,
                    f1_for_val))
            print(" ")


        label_array_for_test, prediction_array_for_test, possibility_array_for_test, total_loss_for_test, auc_for_test, acc_for_test, f1_for_test = val_test_block(
                                                                                                                                                                    model, 
                                                                                                                                                                    loss_fun, 
                                                                                                                                                                    device, 
                                                                                                                                                                    data_for_test, 
                                                                                                                                                                    label_for_test
                                                                                                                                                                  )
        all_fold_auc.append(auc_for_test)
        all_fold_acc.append(acc_for_test)
        all_fold_f1.append(f1_for_test)
        all_pre_result.append(pre_result)
        print("Fold no: " + str(repeat_num_temp))
        print('auc test: ' + str(auc_for_test))
        print('acc test: ' + str(acc_for_test))
        print('f1 test:' + str(f1_for_test))

    print('all acc test:')
    for r in all_fold_acc:
        print(r)
    print('mean acc test: ', np.mean(np.array(all_fold_acc)))
    print('stdev acc test: ', stdev(all_fold_acc))
    print(" ")

    print('all f1 test:')
    for r in all_fold_f1:
        print(r)
    print('mean f1 test', np.mean(np.array(all_fold_f1)))
    print('stdev f1 test: ', stdev(all_fold_f1))
    print(" ")

    print('all auc test:')
    for r in all_fold_auc:
        print(r)
    print('mean auc test', np.mean(np.array(all_fold_auc)))
    print('stdev auc test: ', stdev(all_fold_auc))
    print(" ")


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_data_path", type=str, default="data/h2graph_data/kimia_aug_combine_BRCA_10x20x_512pixel_tree_8nb_gnn_data.pkl", help="Path of training data")
    parser.add_argument("--patient_and_label_path", type=str, default="data/h2graph_data/labels/real_BRCA_patient_label.pkl",
                        help="Path of patient and label data")
    parser.add_argument("--repeat_num", type=int, default=5, help="Number of repetitions of the experiment")
    parser.add_argument("--divide_seed", type=int, default=0, help="Data division seed")
    parser.add_argument("--drop_out_ratio", type=float, default=0.3, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate of model training")
    parser.add_argument("--epochs", type=int, default=60, help="Cycle times of model training")
    parser.add_argument("--batch_size", type=int, default=1, help="Data volume of model training once")
    parser.add_argument("--saved_model_path", type=str, default="your save path",
                        help="Save the path prefix of the model")
    parser.add_argument("--out_classes", type=int, default=256, help="Model middle dimension")
    parser.add_argument("--number_of_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--pool1_ratio", type=float, default=0.1, help="Proportion of the first pool")
    parser.add_argument("--pool2_ratio", type=float, default=4, help="Proportion of the second pool")
    parser.add_argument("--pool3_ratio", type=float, default=4, help="Proportion of the third pool")
    parser.add_argument("--mpool_method", type=str, default="global_mean_pool", help="Global pool method")

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        args = get_params()
        main(args)
    except Exception as exception:
        raise
