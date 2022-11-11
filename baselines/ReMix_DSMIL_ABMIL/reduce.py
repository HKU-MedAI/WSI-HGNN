import argparse
import os

import numpy as np
from tqdm import tqdm

from tools.clustering import Kmeans


def reduce(args, train_list):
    prototypes = []
    semantic_shifts = []
    real_train_list = []
    for feat_pth in tqdm(train_list):
        try:
            feats = np.load(feat_pth)
            feats = np.ascontiguousarray(feats, dtype=np.float32)
            kmeans = Kmeans(k=args.num_prototypes, pca_dim=-1)
            kmeans.cluster(feats, seed=66)  # for reproducibility
            assignments = kmeans.labels.astype(np.int64)
            # compute the centroids for each cluster
            centroids = np.array([np.mean(feats[assignments == i], axis=0)
                                for i in range(args.num_prototypes)])

            # compute covariance matrix for each cluster
            covariance = np.array([np.cov(feats[assignments == i].T)
                                for i in range(args.num_prototypes)])
            # the semantic shift vectors are enough.
            semantic_shift_vectors = []
            for cov in covariance:
                semantic_shift_vectors.append(
                    # sample shift vector from zero-mean multivariate Gaussian distritbuion N(0, cov)
                    np.random.multivariate_normal(np.zeros(cov.shape[0]), cov,
                                                size=args.num_shift_vectors))

            semantic_shift_vectors = np.array(semantic_shift_vectors)
            prototypes.append(centroids)
            semantic_shifts.append(semantic_shift_vectors)
            del feats
            real_train_list.append(feat_pth)
        except:
            print("failed to reduce train id: " + feat_pth)
    prototypes = np.array(prototypes)
    semantic_shifts = np.array(semantic_shifts)
    os.makedirs(f'datasets/{args.dataset}/remix_processed', exist_ok=True)
    np.save(f'datasets/{args.dataset}/remix_processed/train_bag_feats_proto_{args.num_prototypes}_base.npy', prototypes)
    np.save(f'datasets/{args.dataset}/remix_processed/train_bag_feats_shift_{args.num_prototypes}_base.npy', semantic_shifts)

    gt_labels_path = f'datasets/{args.dataset}/{args.dataset}_patient_label_v2.pkl'
    coad_wsi_list = np.load(gt_labels_path, allow_pickle=True)
    train_list = [os.path.basename(x).split('.')[0] for x in real_train_list]
    train_labels = []
    train_list_txt = open(f'datasets/{args.dataset}/remix_processed/train_list_base.txt', 'w')
    for i in train_list:
        classname = '0-normal-npy' if coad_wsi_list[i] == 0 else '1-tumor-npy'
        train_list_txt.write(os.path.join(f'datasets/{args.dataset}', classname, i+'.npy')+','+str(coad_wsi_list[i])+'\n')
        train_labels.append(coad_wsi_list[i])
    np.save(f'datasets/{args.dataset}/remix_processed/train_bag_labels_base.npy', train_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--dataset', type=str, default='BRCA')
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--num_shift_vectors', type=int, default=200)
    args = parser.parse_args()
    train_list = f'datasets/{args.dataset}/remix_processed/train_list_base.txt'
    train_list = open(train_list, 'r').readlines()
    train_list = [x.split(',')[0] for x in train_list]  # file names
    reduce(args, train_list)
