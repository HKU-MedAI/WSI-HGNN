import os
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from skimage.io import imsave
import matplotlib
import matplotlib.cm as cm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score
from xml.dom import minidom

from .evaluator import Evaluator
from data import C16EvalDataset
from parser import parse_gnn_model
from explainers import GNNExplainer, GemExplainer, HetGemExplainer


class ExplainGraph(Evaluator):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

        # Load testing graphs into graph dataloader
        self.name = self.config_data["dataset"]
        eval_path = self.config_data["eval_path"]
        self.patches_path = self.config_data["patches_path"]
        self.wsi_path = self.config_data["wsi_path"]
        self.explain_path = self.config_eval["explain_path"]
        self.annot_path = self.config_eval["annotation_path"]
        if not Path(self.explain_path).exists():
            Path(self.explain_path).mkdir(parents=True)

        self.eval_data = C16EvalDataset(eval_path, self.annot_path)

        # Load trained GNN model
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)
        state_dict = self.checkpoint_manager.load_model()
        self.gnn.load_state_dict(state_dict)
        self.gnn.eval()
        self.n_hops = self.gnn.n_layers - 1

        # WSI info
        self.level = self.config_eval["level"]  # base level 1
        self.base_patch_size = self.config_eval["patch_size"]
        self.patch_size = self.config_eval["patch_size"] // (2 ** (self.level - 1))

        # Name of explainer
        self.explainer_name = self.config_eval["explainer_name"]

    def get_magnified_image(self, name):
        # Initialize extractor to extract patches
        if self.name == "COAD":
            path = self.wsi_path + name + ".svs"
        else:
            path = self.wsi_path + name + ".tif"

        print("Whole slide image name \t %s" % name)
        wsi = OpenSlide(path)
        print("\t Image dimensions @ level 0 \t", wsi.dimensions)
        dim = wsi.level_dimensions[self.level]
        print("\t Image dimensions @ level " + str(self.level) + "\t", dim)
        img = wsi.get_thumbnail(dim)
        return img, wsi

    def get_patch_coords(self, name, dz):
        mag_factor = 2 ** (self.level - 1)

        coords_dir = Path(self.patches_path) / name
        coords = [p for p in coords_dir.iterdir()]
        coords = [p.parts[-1][:-5].split("_") for p in coords]
        coords = [(int(x), int(y))for x, y in coords]
        coords = [(dz._z_from_t(x) // mag_factor, dz._z_from_t(y) // mag_factor) for x, y in coords]  # log2(40/20) = 1

        return coords

    def get_ground_truths(self, xml_path, patches_coords):
        """
        Ground truth path in xml
        :param gt_path:
        :return:
        """
        polygons = minidom.parse(xml_path).getElementsByTagName("Coordinates")
        polygons_out = []
        polygon_coords = []
        for p in polygons:
            coords = []
            for c in p.childNodes:
                if c.attributes:
                    x_coords = c.attributes["X"].value
                    y_coords = c.attributes["Y"].value
                else:
                    continue
                coords.append((float(x_coords), float(y_coords)))
            coords = np.stack(coords)
            polygon_coords.append(coords)
            polygons_out.append(Polygon(coords))

        gt_labels = []
        for c in patches_coords:
            # Get center
            mag_factor = 2 ** (self.level)
            s = self.base_patch_size * 2 // 2  # Patch size at level 0
            c = (k * mag_factor + s for k in c)
            point = Point(c)
            flag = False
            for p in polygons_out:
                if p.contains(point):
                    flag = True
            if flag is True:
                gt_labels.append(1)
            else:
                gt_labels.append(0)

        return gt_labels, polygon_coords


    @staticmethod
    def color_map_color(value, cmap_name='Wistia', vmin=0, vmax=1):
        # norm = plt.Normalize(vmin, vmax)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)  # PiYG
        color = cmap(norm(value))[:, :3]  # will return rgba, we take only first 3 so we get rgb
        return color

    def visualize(self, node_feat_mask, wsi_name, patches_coords, poly_coords, img):

        output_name = os.path.join(self.explain_path, wsi_name + ".png")
        imsave(output_name, np.asarray(img))
        img = np.asarray(img)

        colours = self.color_map_color(node_feat_mask)

        for idx, (bbox, cl) in enumerate(zip(patches_coords, colours)):
            cl = [c * 255 for c in cl]
            s = self.patch_size
            img = cv2.rectangle(img, (bbox[0] + s, bbox[1]), (bbox[0], bbox[1] + s), cl, cv2.FILLED)

        for coords in poly_coords:
            mag_fac = 2 ** (self.level)
            coords = coords.reshape((-1, 1, 2)) / mag_fac
            img = cv2.polylines(img, np.int32([coords]), False, (255, 0, 0), thickness=4)

        output_name = os.path.join(self.explain_path + wsi_name + ".jpeg")
        imsave(output_name, np.asarray(img))

    def eval(self):
        auc_list = []
        for idx in range(len(self.eval_data)):
            path = self.eval_data.graph_paths[idx]
            graph, xml_path, label = self.eval_data[idx]
            graph = graph.to(self.device)
            label = label + torch.zeros(1, dtype=torch.long).to(self.device)
            wsi_name = Path(path).parts[-1][:-4]

            if self.explainer_name == "GNNExplainer":
                explainer = GNNExplainer(graph, self.gnn, num_hops=self.n_hops)
                subgraph, node_mask = explainer.explain_node(node_idx=None)  # None: Graph classification
            elif self.explainer_name == "GemExplainer":
                if graph.is_homogeneous:
                    explainer = GemExplainer(graph, self.gnn, label)
                else:
                    explainer = HetGemExplainer(graph, self.gnn, label)
                node_mask = explainer.explain_node()
            else:
                raise NotImplementedError("This Explainer is not implemented")

            # Perform pixel_wise evaluation
            img, wsi = self.get_magnified_image(wsi_name)
            dz = DeepZoomGenerator(wsi, self.base_patch_size, overlap=0)
            patches_coords = self.get_patch_coords(wsi_name, dz)

            labels, poly_coords = self.get_ground_truths(xml_path, patches_coords)

            fpr, tpr, thresholds = roc_curve(labels, node_mask)
            aucroc = auc(fpr, tpr)
            auc_list.append(aucroc)

            self.visualize(node_mask, wsi_name, patches_coords, poly_coords, img)
            print(f"Mean AUCROC: {np.nanmean(auc_list)}")
