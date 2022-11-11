import os
import pickle
import sys
from pathlib import Path
from importlib import import_module

from typing import OrderedDict
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import numpy as np
from termcolor import colored
from itertools import chain
import nmslib

# Graph Network Packages
import dgl
from scipy.stats import pearsonr

from .extractor import Extractor
from efficientnet_pytorch import EfficientNet

from data import PatchData

'''
Graph construction v2 for new format of patches

Node types:
From PanNuke dataset
0) No-label '0'
1) Neoplastic '1'
2) Inflammatory '2'
3) Connective '3'
4) Dead '4'
5) Non-Neoplastic Epithelial '5'

Edge types:
0 or 1 based on Personr correlation between nodes
'''


class Hnsw:
    """
    KNN model cloned from https://github.com/mahmoodlab/Patch-GCN/blob/master/WSI-Graph%20Construction.ipynb
    """

    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                    "%s: Detect checkpoint saved in data-parallel mode."
                    " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


class Hovernet_infer:
    """ Run HoverNet inference """

    def __init__(self, config, dataloader):
        self.dataloader = dataloader

        method_args = {
            'method': {
                'model_args': {'nr_types': config['nr_types'], 'mode': config['mode'], },
                'model_path': config['hovernet_model_path'],
            },
            'type_info_path': config['type_info_path'],
        }
        run_args = {
            'batch_size': config['batch_size'],
        }

        model_desc = import_module("models.hovernet.net_desc")
        model_creator = getattr(model_desc, "create_model")
        net = model_creator(**method_args['method']["model_args"])
        saved_state_dict = torch.load(method_args['method']["model_path"])["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)
        net.load_state_dict(saved_state_dict, strict=False)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        module_lib = import_module("models.hovernet.run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_infer = lambda input_batch: run_step(input_batch, net)

    def predict(self):
        output_list = []
        features_list = []
        for idx, data in enumerate(self.dataloader):
            data = data.permute(0, 3, 2, 1)
            output, features = self.run_infer(data)
            # curr_batch_size = output.shape[0]
            # output = np.split(output, curr_batch_size, axis=0)[0].flatten()
            features_list.append(features)
            for out in output:
                if out.any() == 0:
                    output_list.append(0)
                else:
                    out = out[out != 0]
                    max_occur_node_type = np.bincount(out).argmax()
                    output_list.append(max_occur_node_type)

        return output_list, np.concatenate(features_list)


class fully_connected(nn.Module):
    """docstring for BottleNeck"""

    def __init__(self, model, num_ftrs, num_classes):
        super(fully_connected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        return out_1, out_3


class KimiaNet_infer:
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

        self.model = torchvision.models.densenet121(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.features = nn.Sequential(self.model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        num_ftrs = self.model.classifier.in_features
        self.model_final = fully_connected(self.model.features, num_ftrs, 512)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_final = nn.DataParallel(self.model_final)
        self.model_final = self.model_final.to(self.device)
        # self.model_final = nn.DataParallel(self.model_final)
        state_dict = torch.load(self.config['kimianet_model_path'])
        sd = self.model_final.state_dict()
        for (k, v), ky in zip(state_dict.items(), sd.keys()):
            sd[ky] = v
        self.model_final.load_state_dict(sd)

    def predict(self):
        self.model_final.eval()
        features_list = []
        for idx, data in enumerate(self.dataloader):
            # data = data.permute(0, 3, 2, 1)
            data = data.to(self.device)
            output1, _ = self.model_final(data)
            output_1024 = output1.cpu().detach().numpy()
            features_list.append(output_1024)
        return np.concatenate(features_list)


class EfficientNet_infer:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_final = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1024).to(self.device)

    def predict(self):
        self.model_final.eval()
        features_list = []
        for idx, data in enumerate(self.dataloader):
            # data = data.permute(0, 3, 2, 1)
            data = data.to(self.device)
            output1 = self.model_final(data)
            output_1024 = output1.cpu().detach().numpy()
            features_list.append(output_1024)
        return np.concatenate(features_list)


class GraphConstructor:
    def __init__(self, config: OrderedDict, hovernet_config: OrderedDict, kimianet_config: OrderedDict, wsi_data):
        self.config = config
        self.hovernet_config = hovernet_config
        self.kimianet_config = kimianet_config
        self.wsi_data = wsi_data

        self.radius = self.config['radius']
        self.knn_model = Hnsw(space='l2')

        patch_path = Path(wsi_data)
        patch_dataset = PatchData(patch_path)
        dataloader = data.DataLoader(
            patch_dataset,
            num_workers=0,
            batch_size=8,
            shuffle=False
        )

        self.encoder_name = config['encoder_name']
        node_type_dir = config["node_type_dir"]
        hovernet_model = Hovernet_infer(self.hovernet_config, dataloader)
        if node_type_dir is None or self.encoder_name == "hover":
            self.node_type, self.features = hovernet_model.predict()
        elif node_type_dir:
            head, tail = os.path.split(wsi_data)
            node_type_file = os.path.join(node_type_dir + tail + '.pkl')
            with open(node_type_file, "rb") as f:
                self.node_type = pickle.load(f)

        if self.encoder_name == "kimia":
            print("Use KimiaNet pretrained model!")
            kimia_model = KimiaNet_infer(self.kimianet_config, dataloader)
            self.features = kimia_model.predict()
        elif self.encoder_name == "efficientnet-b4":
            encoder = EfficientNet_infer(dataloader)
            self.features = encoder.predict()

    def construct_graph(self):

        ####################
        # Step 1 Fit the coordinate with KNN and construct edges
        ####################
        # Number of patches
        n_patches = self.features.shape[0]

        # Construct graph using spatial coordinates
        self.knn_model.fit(self.features)

        a = np.repeat(range(n_patches), self.radius - 1)
        b = np.fromiter(
            chain(
                *[self.knn_model.query(self.features[v_idx], topn=self.radius)[1:] for v_idx in range(n_patches)]
            ), dtype=int
        )
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Create edge types
        edge_type = []
        edge_sim = []
        for (idx_a, idx_b) in zip(a, b):
            metric = pearsonr
            corr = metric(self.features[idx_a], self.features[idx_b])[0]
            edge_type.append(1 if corr > 0 else 0)
            edge_sim.append(corr)

        # Construct dgl heterogeneous graph
        graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
        graph.ndata.update({'_TYPE': torch.tensor(self.node_type)})
        self.features = torch.tensor(self.features, device='cpu').float()
        # self.patches_coords = torch.tensor(self.patches_coords, device='cpu').float()
        graph.ndata['feat'] = self.features
        # graph.ndata['patches_coords'] = self.patches_coords
        graph.edata.update({'_TYPE': torch.tensor(edge_type)})
        graph.edata.update({'sim': torch.tensor(edge_sim)})
        het_graph = dgl.to_heterogeneous(
            graph,
            [str(t) for t in range(self.config["n_node_type"])],
            ['neg', 'pos']
        )

        homo_graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
        homo_graph.ndata['feat'] = self.features
        # homo_graph.ndata['patches_coords'] = self.patches_coords

        return het_graph, homo_graph, self.node_type
