import argparse
import glob
import os
import threading
from tkinter import N
import urllib.request
import zipfile
from os.path import join

from matplotlib.style import available

# import cv2
import dgl
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from skimage import img_as_ubyte, io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print('downloading dataset')
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def unzip_data(zip_path, data_path):
    os.makedirs(data_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)


def process_COAD():
    dataset_pth = 'datasets/COAD/'
    graph_path = os.path.join(dataset_pth, 'COAD_hover_lv0/homogeneous')
    coad_wsi_list = np.load(os.path.join(dataset_pth, 'COAD_patient_label.pkl'), allow_pickle=True)

    available_normal_list = []
    available_tumor_list = []
    for token in ['0-normal', '1-tumor']:
        os.makedirs(f'{dataset_pth}/{token}-npy', exist_ok=True)
        class_id = 0 if token == '0-normal' else 1
        wsi_per_class_id_list = [k for k, v in  coad_wsi_list.items() if v == class_id]
        for wsi in tqdm(wsi_per_class_id_list, desc=f'converting COAD data to npy, {token}'):
            try:
                with open(os.path.join(graph_path, wsi+'.pkl'),'rb') as f:
                    graph = pickle.load(f)
                    feats = graph.ndata['feat'].numpy() # torch.Size([patches, 1024])
                np.save(os.path.join(dataset_pth, f'{token}-npy',wsi+'.npy'), feats)
                if class_id == 0:
                    available_normal_list.append(wsi)
                else:
                    available_tumor_list.append(wsi)
            except: 
                print("graph not found")

    print("Split COAD data ...")
    # Split 80/20
    train_normal_data = available_normal_list[:int((len(available_normal_list) + 1)*.80)]
    test_normal_data = available_normal_list[int((len(available_normal_list) + 1)*.80):]
    train_tumor_data = available_tumor_list[:int((len(available_tumor_list) + 1)*.80)]
    test_tumor_data = available_tumor_list[int((len(available_tumor_list) + 1)*.80):]

    train_data = train_normal_data + train_tumor_data
    random.shuffle(train_data)
    test_data = test_normal_data + test_tumor_data
    random.shuffle(test_data)

    os.makedirs('datasets/COAD/remix_processed', exist_ok=True)
    train_list_txt = open('datasets/COAD/remix_processed/train_list.txt', 'w')
    test_list_txt = open('datasets/COAD/remix_processed/test_list.txt', 'w')
    train_labels, test_labels = [], []

    for i in tqdm(train_data, desc='Train COAD data'):
        class_name = '0-normal-npy' if coad_wsi_list[i] == 0 else '1-tumor-npy'
        train_list_txt.write(os.path.join(dataset_pth, class_name, i+'.npy')+','+str(coad_wsi_list[i])+'\n')
        train_labels.append(coad_wsi_list[i])

    for i in tqdm(test_data, desc='Test COAD data'):
        class_name = '0-normal-npy' if coad_wsi_list[i] == 0 else '1-tumor-npy'
        test_list_txt.write(os.path.join(dataset_pth, class_name, i+'.npy')+','+str(coad_wsi_list[i])+'\n')
        test_labels.append(coad_wsi_list[i])

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    np.save('datasets/COAD/remix_processed/test_bag_labels.npy', test_labels)
    np.save('datasets/COAD/remix_processed/train_bag_labels.npy', train_labels)
    print('done')


def process_BRCA():
    
    dataset_pth = 'datasets/BRCA/'
    graph_path = os.path.join(dataset_pth, 'BRCA_kimia_lv0/homogeneous')
    coad_wsi_list = np.load(os.path.join(dataset_pth, 'BRCA_patient_label_v2.pkl'), allow_pickle=True)

    available_normal_list = []
    available_tumor_list = []
    for token in ['0-normal', '1-tumor']:
        os.makedirs(f'{dataset_pth}/{token}-npy', exist_ok=True)
        class_id = 0 if token == '0-normal' else 1
        wsi_per_class_id_list = [k for k, v in  coad_wsi_list.items() if v == class_id]
        for wsi in tqdm(wsi_per_class_id_list, desc=f'converting BRCA data to npy, {token}'):
            try:
                with open(os.path.join(graph_path, wsi+'.pkl'),'rb') as f:
                    graph = pickle.load(f)
                    feats = graph.ndata['feat'].numpy() # torch.Size([patches, 1024])
                np.save(os.path.join(dataset_pth, f'{token}-npy',wsi+'.npy'), feats)
                if class_id == 0:
                    available_normal_list.append(wsi)
                else:
                    available_tumor_list.append(wsi)
            except: 
                print("graph not found")

    print("Split BRCA data ...")
    # Split 80/20
    train_normal_data = available_normal_list[:int((len(available_normal_list) + 1)*.80)]
    test_normal_data = available_normal_list[int((len(available_normal_list) + 1)*.80):]
    train_tumor_data = available_tumor_list[:int((len(available_tumor_list) + 1)*.80)]
    test_tumor_data = available_tumor_list[int((len(available_tumor_list) + 1)*.80):]

    train_data = train_normal_data + train_tumor_data
    random.shuffle(train_data)
    test_data = test_normal_data + test_tumor_data
    random.shuffle(test_data)

    os.makedirs('datasets/BRCA/remix_processed', exist_ok=True)
    train_list_txt = open('datasets/BRCA/remix_processed/train_list.txt', 'w')
    test_list_txt = open('datasets/BRCA/remix_processed/test_list.txt', 'w')
    train_labels, test_labels = [], []

    for i in tqdm(train_data, desc='Train BRCA data'):
        class_name = '0-normal-npy' if coad_wsi_list[i] == 0 else '1-tumor-npy'
        train_list_txt.write(os.path.join(dataset_pth, class_name, i+'.npy')+','+str(coad_wsi_list[i])+'\n')
        train_labels.append(coad_wsi_list[i])

    for i in tqdm(test_data, desc='Test BRCA data'):
        class_name = '0-normal-npy' if coad_wsi_list[i] == 0 else '1-tumor-npy'
        test_list_txt.write(os.path.join(dataset_pth, class_name, i+'.npy')+','+str(coad_wsi_list[i])+'\n')
        test_labels.append(coad_wsi_list[i])

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    np.save('datasets/BRCA/remix_processed/test_bag_labels.npy', test_labels)
    np.save('datasets/BRCA/remix_processed/train_bag_labels.npy', train_labels)
    print('done')


def thres_saturation(img, t=15):
    # typical t = 15
    _img = np.array(img)
    _img = rgb2hsv(_img)
    h, w, c = _img.shape
    sat_img = _img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t


def slide_to_patch_jpeg(out_base, img_slides, patch_size=224):
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(os.path.sep)[-1].split('.png')[0].replace(' ', '_')
        img_class = img_slide.split(os.path.sep)[-2]
        bag_path = join(out_base, img_class, img_name)
        os.makedirs(bag_path, exist_ok=True)
        img = Image.open(img_slide).convert('RGB')
        w, h = img.size
        step_j = h // 8
        step_i = w // 8
        for i in range(8):
            for j in range(8):
                roi = img.crop((int(j * step_j), int(i * step_i), int((j + 1) * step_j), int((i + 1) * step_i)))
                if thres_saturation(roi, 30):
                    resized = cv2.resize(np.array(roi), (patch_size, patch_size))
                    patch_name = "{}_{}".format(i, j)
                    io.imsave(join(bag_path, patch_name + ".jpg"), img_as_ubyte(resized))


def split_unitopatho(download_dir):
    raw_image_pth = 'datasets/Unitopatho/raw'
    print('gathering all patches....')
    all_patches = glob.glob(f'{raw_image_pth}/*/*/*.jpg')
    print('gathered', len(all_patches), 'samples.')
    ############# For OpenSelfSup pre-training ###########################
    os.makedirs('datasets/Unitopatho/meta', exist_ok=True)
    train_txt = open('datasets/Unitopatho/meta/train.txt', 'w')
    test_txt = open('datasets/Unitopatho/meta/test.txt', 'w')
    # These files are included in the downloaded files.
    train_wsi_txt = open(f'{download_dir}/train_wsi.txt', 'r').readlines()
    test_wsi_txt = open(f'{download_dir}/test_wsi.txt', 'r').readlines()
    train_wsi_txt = [l.strip() for l in train_wsi_txt]
    test_wsi_txt = [l.strip() for l in test_wsi_txt]
    
    top_label_mapping = {'HP': 0, 'NORM': 1, 'TA.HG': 2, 'TA.LG': 3, 'TVA.HG': 4, 'TVA.LG': 5}
    for pth in tqdm(all_patches, desc='creating Unitopatho meta files'):
        pth_parsed = pth.split('/')
        top_label = top_label_mapping[pth_parsed[-3]]
        wsi_name = pth_parsed[-2].split('.ndpi')[0].replace('_', ' ')
        if wsi_name in train_wsi_txt:
            train_txt.write(f"{'/'.join(pth_parsed[-3:])}\n")
        elif wsi_name in test_wsi_txt:
            test_txt.write(f"{'/'.join(pth_parsed[-3:])}\n")
        else:
            print('Error:', wsi_name)
    ######################################################################

    ################# Extract bag list ###################################
    train_bag_pth = 'datasets/Unitopatho/meta/wsi_list/train'
    test_bag_pth = 'datasets/Unitopatho/meta/wsi_list/test'
    os.makedirs(train_bag_pth, exist_ok=True)
    os.makedirs(test_bag_pth, exist_ok=True)
    for i, pth in tqdm(enumerate(all_patches), total=len(all_patches),
                       desc='creating Unitopatho bag list'):
        pth_parsed = pth.split('/')
        top_label = top_label_mapping[pth_parsed[-3]]
        wsi_name = pth_parsed[-2].split('.ndpi')[0].replace('_', ' ')
        if wsi_name in train_wsi_txt:
            if i == 0:  # force rewrite
                patch_list = open(f'{train_bag_pth}/{wsi_name}_{top_label}.txt', 'w')
            else:
                patch_list = open(f'{train_bag_pth}/{wsi_name}_{top_label}.txt', 'a')
        elif wsi_name in test_wsi_txt:
            if i == 0:  # force rewrite
                patch_list = open(f'{test_bag_pth}/{wsi_name}_{top_label}.txt', 'w')
            else:
                patch_list = open(f'{test_bag_pth}/{wsi_name}_{top_label}.txt', 'a')
        else:
            print('Error', wsi_name)
            exit()
        file_pth = '/'.join(pth_parsed[-3:])
        patch_list.write(f'{file_pth}\n')
    ######################################################################

    ################# Gather final list ##################################
    os.makedirs('datasets/Unitopatho/remix_processed', exist_ok=True)
    train_list_txt = open('datasets/Unitopatho/remix_processed/train_list.txt', 'w')
    test_list_txt = open('datasets/Unitopatho/remix_processed/test_list.txt', 'w')
    train_files = glob.glob(f'{train_bag_pth}/*.txt')
    test_files = glob.glob(f'{test_bag_pth}/*.txt')
    train_labels, test_labels = [], []
    for f_name in train_files:
        f_name = f_name.split('/')[-1].replace('txt', 'npy')
        label = f_name.split('_')[-1].split('.')[0]
        train_labels.append(int(label))
        train_list_txt.write(f"datasets/Unitopatho/features/train_npy/{f_name},{label}\n")
        
    for f_name in test_files:
        f_name = f_name.split('/')[-1].replace('txt', 'npy')
        label = f_name.split('_')[-1].split('.')[0]
        test_labels.append(int(label))
        test_list_txt.write(f"datasets/Unitopatho/features/test_npy/{f_name},{label}\n")
    
    train_labels, test_labels = np.array(train_labels), np.array(test_labels)
    np.save('datasets/Unitopatho/remix_processed/train_bag_labels.npy', train_labels)
    np.save('datasets/Unitopatho/remix_processed/test_bag_labels.npy', test_labels)
    ######################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset', type=str, default='COAD', choices=['Unitopatho', 'COAD', 'BRCA'])
    parser.add_argument('--task', type=str, nargs="+",
                        default=['download', 'convert', 'split'],
                        choices=['download', 'convert', 'split', 'crop'])
    parser.add_argument('--num_threads', type=int, default=64, help='Number of threads for parallel processing')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch size')
    args = parser.parse_args()

    if args.dataset == 'COAD':
         print('Convert and split COAD datasets (pre-computed features)')
         process_COAD()

    elif args.dataset == 'BRCA':
        print('Convert and split BRCA datasets (pre-computed features)')
        process_BRCA()
            
    elif args.dataset == 'Unitopatho':
        # Please change the download_dir to your own directory.
        download_dir = ''
        
        if args.task[0] == 'crop':
            print('Cropping patches, this could take a while for big dataset, please be patient')
            # path to target folder
            out_base = 'datasets/Unitopatho/raw'
            os.makedirs(out_base, exist_ok=True)
            # we use the resolution of 800
            data_dir = f'{download_dir}/800'
            all_slides = glob.glob(join(data_dir, '*/*.png'))
            each_thread = int(np.floor(len(all_slides) / args.num_threads))
            threads = []
            for i in range(args.num_threads):
                if i < (args.num_threads - 1):
                    t = threading.Thread(target=slide_to_patch_jpeg,
                                         args=(out_base, all_slides[each_thread * i: each_thread * (i + 1)], args.patch_size))
                else:
                    t = threading.Thread(target=slide_to_patch_jpeg,
                                         args=(out_base, all_slides[each_thread * i:], args.patch_size))
                threads.append(t)

            for thread in threads:
                thread.start()
            # wait until all threads finish
            for thread in threads:
                thread.join()
            args.task.pop(0)
            
        if args.task[0] == 'split':
            split_unitopatho(download_dir)
            args.task.pop(0)
