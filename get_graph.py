import os
import sys
import argparse
import pickle
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from math import floor
from collections import OrderedDict
from random import shuffle

import glob

from globals import *
from construct_graph import GraphConstructor


def ordered_yaml():
    """
    yaml orderedDict support
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def randomize_files(file_list):
    shuffle(file_list)


def get_training_and_testing_sets(file_list, split):
    split_index = floor(len(file_list) * split)
    train_files = file_list[:split_index]
    test_files = file_list[split_index:]
    return train_files, test_files


def COAD_trainval(config):
    normal_path = './data/biomedical_data/normal_list.txt'
    graph_list = glob.glob(config['out_dir']+'/homogeneous/*.pkl')
    with open(normal_path) as f:
        # List of path to normal images
        normal_list = [l.strip() for l in f.readlines()]
    normal_graph_list = []
    for normal in normal_list:
        graphs = glob.glob(config['out_dir']+"/homogeneous/"+normal+"*.pkl")
        for graph in graphs:
            normal_graph_list.append(graph)
    print("Total normal graph: " + str(len(normal_graph_list)))

    graph_list_ = list(set(graph_list).difference(set(normal_graph_list)))

    if len(normal_graph_list) + len(graph_list_) != len(graph_list) :
        print("removed graph number != total normal graph!!")
        sys.exit()

    randomize_files(normal_graph_list)
    randomize_files(graph_list_)
    
    train_list, testval_list = get_training_and_testing_sets(graph_list_, 0.8)
    test_list, val_list = get_training_and_testing_sets(testval_list, 0.5)
    train_normal_list, testval_normal_list = get_training_and_testing_sets(normal_graph_list, 0.8)
    test_normal_list, val_normal_list = get_training_and_testing_sets(testval_normal_list, 0.5)
    train_list = train_list + train_normal_list
    test_list = test_list + test_normal_list
    val_list = val_list + val_normal_list

    return train_list, val_list, test_list


def BRCA_trainval(config):
    normal_path = 'data/biomedical_data/normal_list_BRCA.txt'
    graph_list = glob.glob(config['out_dir']+'/homogeneous/*.pkl')
    with open(normal_path) as f:
        normal_list = [l.strip() for l in f.readlines()]
    normal_graph_list = []
    for normal in normal_list:
        graphs = glob.glob(config['out_dir']+"/homogeneous/"+normal+"*.pkl")
        for graph in graphs:
            normal_graph_list.append(graph)

    print("Total normal graph: "+ str(len(normal_graph_list)))

    graph_list_ = list(set(graph_list).difference(set(normal_graph_list)))

    if len(normal_graph_list) + len(graph_list_) != len(graph_list):
        print("Removed graph number != total normal graph!")
        sys.exit()

    randomize_files(normal_graph_list)
    randomize_files(graph_list_)

    train_list, testval_list = get_training_and_testing_sets(graph_list_, 0.8)
    test_list, val_list = get_training_and_testing_sets(testval_list, 0.5)
    train_normal_list, testval_normal_list = get_training_and_testing_sets(normal_graph_list, 0.8)
    test_normal_list, val_normal_list = get_training_and_testing_sets(testval_normal_list, 0.5)
    train_list = train_list + train_normal_list
    test_list = test_list + test_normal_list
    val_list = val_list + val_normal_list

    return train_list, val_list, test_list


def COAD_staging_train_val(config):
    normal_path = 'data/biomedical_data/normal_list.txt'
    staging_path = 'data/clinical_data/staging.txt'
    with open(normal_path) as f:
        # List of path to normal images
        normal_list = [l.strip() for l in f.readlines()]
    with open(staging_path) as f:
        mapping = [l.strip().split(sep="\t") for l in f.readlines()]
        mapping = {k: v for k, v in mapping}
    all_paths = glob.glob(config['out_dir']+'/homogeneous/*.pkl')

    # Remove graphs that have no types
    graphs = []
    for p in all_paths:
        pos = p.find("TCGA")
        if p[pos:pos+16] in normal_list:
            continue
        try:
            if mapping[p[pos:pos+12]] not in ['Stage I', 'Stage IIIB', 'Stage IIA', 'Stage IV',
                                              'Stage IIB', 'Stage IIIC', 'Stage II', 'Stage IVA',
                                              'Stage IIC', 'Stage III', 'Stage IIIA', 'Stage IVB', 'Stage IA']:
                continue
        except KeyError:
            continue
        graphs.append(p)

    randomize_files(graphs)

    train_list, testval_list = get_training_and_testing_sets(graphs, 0.8)
    test_list, val_list = get_training_and_testing_sets(testval_list, 0.5)

    return train_list, val_list, test_list


def BRCA_staging_train_val(config):
    normal_path = 'data/biomedical_data/normal_list_BRCA.txt'
    staging_path = 'data/clinical_data/staging_BRCA.txt'
    with open(normal_path) as f:
        # List of path to normal images
        normal_list = [l.strip() for l in f.readlines()]
    with open(staging_path) as f:
        mapping = [l.strip().split(sep="\t") for l in f.readlines()]
        mapping = {k: v for k, v in mapping}
    all_paths = glob.glob(config['out_dir']+'/homogeneous/*.pkl')

    # Remove graphs that have no types
    graphs = []
    for p in all_paths:
        pos = p.find("TCGA")
        if p[pos:pos+16] in normal_list:
            continue
        try:
            if mapping[p[pos:pos+12]] not in ['Stage I', 'Stage IIIB', 'Stage IIA', 'Stage IV',
                                              'Stage IIB', 'Stage IIIC', 'Stage II', 'Stage IVA',
                                              'Stage IIC', 'Stage III', 'Stage IIIA', 'Stage IVB', 
                                              'Stage IA', 'Stage IB']:
                continue
        except KeyError:
            continue
        graphs.append(p)

    randomize_files(graphs)

    train_list, testval_list = get_training_and_testing_sets(graphs, 0.8)
    test_list, val_list = get_training_and_testing_sets(testval_list, 0.5)

    return train_list, val_list, test_list

def BRCA_typing_train_val(config):
    normal_path = 'data/biomedical_data/normal_list_BRCA.txt'
    staging_path = 'data/clinical_data/typing_BRCA.txt'
    with open(normal_path) as f:
        # List of path to normal images
        normal_list = [l.strip() for l in f.readlines()]
    with open(staging_path) as f:
        mapping = [l.strip().split(sep="\t") for l in f.readlines()]
        mapping = {k: v for k, v in mapping}
    all_paths = glob.glob(config['out_dir']+'/homogeneous/*.pkl')

    # Remove graphs that have no types
    graphs = []
    for p in all_paths:
        pos = p.find("TCGA")
        if p[pos:pos+16] in normal_list:
            continue
        try:
            if mapping[p[pos:pos+12]] not in ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma']:
                continue
        except KeyError:
            continue
        graphs.append(p)

    randomize_files(graphs)

    train_list, testval_list = get_training_and_testing_sets(graphs, 0.6)
    test_list, val_list = get_training_and_testing_sets(testval_list, 0.7)

    return train_list, val_list, test_list

def camelyon16_trainval(config):
    train_data = ('tumor', 'normal')
    train_list, val_list = [], []
    for type_ in train_data:
        train_list.extend(glob.glob(config['out_dir']+'/homogeneous/'+type_+'*.pkl'))

    test_list = glob.glob(config['out_dir']+'/homogeneous/test*.pkl')
    test_list, val_list = get_training_and_testing_sets(test_list, 0.5)

    return train_list, val_list, test_list


parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")

args = parser.parse_args()

opt_path = args.config
default_config_path = "GraphConstruction/BRCA_HovernetKimia_graph_constructor.yml"
CONSTRUCT = False
GET_TRAINVAL = True

if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path


def main():

    with open(opt_path, mode='r') as f:
            loader, _ = ordered_yaml()
            config = yaml.load(f, loader)
            print(f"Loaded configs from {opt_path}")

    graph_config = config['graph_constructor']
    hovernet_config = config['hovernet_config']
    kimianet_config = config['kimianet_config']

    if CONSTRUCT:
        
        patch_paths = glob.glob(graph_config["patch_path"] + "*/*")

        for i, wsi_input in enumerate(patch_paths):
            print(f"Processing {i+1} / {len(patch_paths)}")
            try:

                head, tail = os.path.split(wsi_input)
                het_output_file = os.path.join(graph_config['out_dir'] + '/heterogeneous/' + tail + '.pkl')
                homo_output_file = os.path.join(graph_config['out_dir'] + '/homogeneous/' + tail + '.pkl')
                node_type_file = os.path.join(graph_config['out_dir'] + '/node_types/' + tail + '.pkl')
                if Path(het_output_file).exists() or Path(homo_output_file).exists():
                    continue

                graph_constructor = GraphConstructor(graph_config, hovernet_config, kimianet_config, wsi_input)
                het_graph, homo_graph, node_type = graph_constructor.construct_graph()

                # Make directory
                if not Path(graph_config['out_dir'] + '/heterogeneous/').exists():
                    Path(graph_config['out_dir'] + '/heterogeneous/').mkdir(parents=True)
                if not Path(graph_config['out_dir'] + '/homogeneous/').exists():
                    Path(graph_config['out_dir'] + '/homogeneous/').mkdir(parents=True)
                if not Path(graph_config['out_dir'] + '/node_types/').exists():
                    Path(graph_config['out_dir'] + '/node_types/').mkdir(parents=True)


                with open(het_output_file, 'wb') as f:
                    pickle.dump(het_graph, f)
                print("Het Graph saved at: " + het_output_file)

                with open(homo_output_file, 'wb') as g:
                    pickle.dump(homo_graph, g)
                print("Homo Graph saved at: " + homo_output_file)

                with open(node_type_file, 'wb') as f:
                    pickle.dump(node_type, f)
                print("Node type saved at: " + node_type_file)

                print(' ')

            except (ValueError, KeyError, IndexError, RuntimeError, FileNotFoundError):
                print("Failed to construct graph, moves to next WSI image")
    
    if GET_TRAINVAL:
        fold = 1
        list_name_classf = f"/list_f{fold}/"
        list_name_staging = f"/list_staging_f{fold}/"
        list_name_typing = f"/list_typing_f{fold}/"
        if graph_config['dataset'] == 'COAD':
            if graph_config['task'] == "cancer classification":
                train_list, val_list, test_list = COAD_trainval(graph_config)
                list_name = list_name_classf
            elif graph_config['task'] == "cancer staging":
                train_list, val_list, test_list = COAD_staging_train_val(graph_config)
                list_name = list_name_staging
            else:
                raise ValueError("No such task")
        elif graph_config['dataset'] == 'camelyon16':
            train_list, val_list, test_list = camelyon16_trainval(graph_config)
            list_name = list_name_classf

        elif graph_config['dataset'] == 'BRCA':
            if graph_config['task'] == "cancer classification":
                train_list, val_list, test_list = BRCA_trainval(graph_config)
                list_name = list_name_classf
            elif graph_config['task'] == "cancer staging":
                train_list, val_list, test_list = BRCA_staging_train_val(graph_config)
                list_name = list_name_staging
            elif graph_config['task'] == "cancer typing":
                train_list, val_list, test_list = BRCA_typing_train_val(graph_config)
                list_name = list_name_typing
            else:
                raise ValueError("No such task")
        else:
            raise ValueError("No such dataset")

        print("number of training data: " + str(len(train_list)))
        print("number of val data: " + str(len(val_list)))
        print("number of test data: " + str(len(test_list)))

        check = input("Proceed? y/n\n")
        if check == 'n':
            sys.exit()

        for graph in ['heterogeneous', 'homogeneous']:
            for types in [['_train', train_list], ['_test', test_list], ['_val', val_list]]:
                if not Path(graph_config['out_dir'] + list_name).exists():
                    Path(graph_config['out_dir'] + list_name).mkdir(parents=True)
                f = open(graph_config['out_dir'] + list_name + graph + types[0] + ".txt", "w+")
                # for i in range(len(train_list)):
                for i in types[1]:
                    head, tail = os.path.split(i)
                    f.write(graph_config['out_dir']+'/'+graph+'/'+tail+"\n")
                f.close()
        print(f"Lists saved in {graph_config['out_dir'] + list_name}")

if __name__ == '__main__':
    main()
