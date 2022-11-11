import joblib as joblib
import torch
import numpy as np
from torch_geometric.data import Data
from numpy import *

patient_and_label = joblib.load("/data/h2graph_data/labels/_patient_label.pkl")
no_sort_t_all_feature = joblib.load("data/h2graph_data/features_embedding/20x_512pixel_Kimia_Simclr_Feature_v2_dict.pkl")

num = list()
for x in no_sort_t_all_feature:
     num.append(len(no_sort_t_all_feature[x]))
print(max(num))
print(min(num))
print(mean(num))

save_path = 'data/h2graph_data/kimia_aug_combine_10x20x_512pixel_tree_8nb_gnn_v2_data.pkl'
# save_dict = dict()
# for x in no_sort_t_all_feature:
#      save_dict[x] = 1
# print(save_dict)

# save_dict = {
#      'TCGA-WK-A8XQ-01Z-00-DX2': 0,
#      'TCGA-DX-A6BE-01Z-00-DX2': 0,
#      'TCGA-Z4-A9VC-01Z-00-DX1': 0,
#      'TCGA-DX-AB2F-01Z-00-DX2': 0,
#      'TCGA-IF-A3RQ-01Z-00-DX1': 0,
#      'TCGA-WK-A8XO-01Z-00-DX2': 0,
#      'TCGA-MO-A47P-01Z-00-DX1': 0,
#      'TCGA-DX-AB2F-01Z-00-DX1': 0,
#      'TCGA-WK-A8XQ-01Z-00-DX4': 0,
#      'TCGA-3B-A9HL-01Z-00-DX1': 0,
#      'TCGA-LI-A67I-01Z-00-DX7': 1,
#      'TCGA-DX-AB32-01Z-00-DX4': 1,
#      'TCGA-DX-A7EU-01Z-00-DX1': 1,
#      'TCGA-DX-A7EU-01Z-00-DX5': 1,
#      'TCGA-DX-A6YR-01Z-00-DX1': 1,
#      'TCGA-LI-A67I-01Z-00-DX8': 1,
#      'TCGA-DX-AB2T-01Z-00-DX5': 1,
#      'TCGA-DX-AB2L-01Z-00-DX6': 1,
#      'TCGA-DX-A6YV-01Z-00-DX1': 1,
#      'TCGA-LI-A67I-01Z-00-DX9': 1,
#      'TCGA-DX-A3U7-01Z-00-DX1': 2,
#      'TCGA-X6-A7WD-01Z-00-DX4': 2,
#      'TCGA-X6-A7WD-01Z-00-DX3': 2,
#      'TCGA-DX-A7EN-01Z-00-DX1': 2,
#      'TCGA-DX-A3U9-01Z-00-DX1': 2,
#      'TCGA-3B-A9HQ-01Z-00-DX1': 2,
#      'TCGA-WK-A8XS-01Z-00-DX4': 2,
#      'TCGA-X6-A7WD-01Z-00-DX5': 2,
#      'TCGA-3B-A9I3-01Z-00-DX1': 2,
#      'TCGA-X9-A971-01Z-00-DX3': 2}

# joblib.dump(save_dict,"./esca_patient_and_label.pkl")


copy_patient_label = patient_and_label

patients = []
for x in patient_and_label:
    patients.append(x)

t_all_feature = {}
for x in patients:
     feature = {}
     for z in no_sort_t_all_feature[x]:
          if z == '-1.jpeg':
               feature[z] = no_sort_t_all_feature[x][z]
     for z in no_sort_t_all_feature[x]:
          if len(z.split('-')) == 6:
               feature[z] = no_sort_t_all_feature[x][z]
     for z in no_sort_t_all_feature[x]:
          if len(z.split('-')) == 2 and z != '-1.jpeg':
               feature[z] = no_sort_t_all_feature[x][z]
     t_all_feature[x] = feature

all_feature = {}
for x in patients:
     feature = []
     for z in t_all_feature[x]:
          if z == '-1':
               feature.append(t_all_feature[x][z])
     for z in t_all_feature[x]:
          if len(z.split('-')) == 6:
               feature.append(t_all_feature[x][z])
     for z in t_all_feature[x]:
          if len(z.split('-')) == 2 and z != '-1':
               feature.append(t_all_feature[x][z])

     all_feature[x] = feature


def get_edge_index_2(id):
     start = []
     end = []

     patch_id = {}
     i = 0
     for x in t_all_feature[id]:
          patch_id[x.split('.')[0]] = i
          i += 1
     #     print(patch_id)

     # Edges between nodes of different resolutions
     for x in patch_id:
          if len(x.split('-')) == 6:
               # edge with -1.jpeg to 5x node
               start.append(patch_id[x])
               end.append(patch_id['-1'])
               end.append(patch_id[x])
               start.append(patch_id['-1'])
               # edge with 5x node to 10x node
               if x.split('-')[0] + '-' + x.split('-')[2] in patch_id:
                    start.append(patch_id[x])
                    end.append(patch_id[x.split('-')[0] + '-' + x.split('-')[2]])
                    end.append(patch_id[x])
                    start.append(patch_id[x.split('-')[0] + '-' + x.split('-')[2]])
               if x.split('-')[1] + '-' + x.split('-')[2] in patch_id:
                    start.append(patch_id[x])
                    end.append(patch_id[x.split('-')[1] + '-' + x.split('-')[2]])
                    end.append(patch_id[x])
                    start.append(patch_id[x.split('-')[1] + '-' + x.split('-')[2]])
               if x.split('-')[0] + '-' + x.split('-')[3] in patch_id:
                    start.append(patch_id[x])
                    end.append(patch_id[x.split('-')[0] + '-' + x.split('-')[3]])
                    end.append(patch_id[x])
                    start.append(patch_id[x.split('-')[0] + '-' + x.split('-')[3]])
               if x.split('-')[1] + '-' + x.split('-')[3] in patch_id:
                    start.append(patch_id[x])
                    end.append(patch_id[x.split('-')[1] + '-' + x.split('-')[3]])
                    end.append(patch_id[x])
                    start.append(patch_id[x.split('-')[1] + '-' + x.split('-')[3]])

     patch_id_5x = {}
     for x in patch_id:
          if len(x.split('-')) == 6:
               patch_id_5x[x.split('-')[-2] + '-' + x.split('-')[-1]] = patch_id[x]
     #     print(patch_id_5x)

     # 5x resolution edge between nodes
     for x in patch_id:
          if len(x.split('-')) == 6:
               i = int(x.split('-')[-2])
               j = int(x.split('-')[-1])
               #add edge with 5x node to its neighbour
               if str(i) + '-' + str(j + 1) in patch_id_5x:
                    start.append(patch_id_5x[str(i) + '-' + str(j)])
                    end.append(patch_id_5x[str(i) + '-' + str(j + 1)])
               if str(i) + '-' + str(j - 1) in patch_id_5x:
                    start.append(patch_id_5x[str(i) + '-' + str(j)])
                    end.append(patch_id_5x[str(i) + '-' + str(j - 1)])
               if str(i + 1) + '-' + str(j) in patch_id_5x:
                    start.append(patch_id_5x[str(i) + '-' + str(j)])
                    end.append(patch_id_5x[str(i + 1) + '-' + str(j)])
               if str(i - 1) + '-' + str(j) in patch_id_5x:
                    start.append(patch_id_5x[str(i) + '-' + str(j)])
                    end.append(patch_id_5x[str(i - 1) + '-' + str(j)])
               if str(i + 1) + '-' + str(j + 1) in patch_id_5x:
                    start.append(patch_id_5x[str(i) + '-' + str(j)])
                    end.append(patch_id_5x[str(i + 1) + '-' + str(j + 1)])
               if str(i - 1) + '-' + str(j + 1) in patch_id_5x:
                    start.append(patch_id_5x[str(i) + '-' + str(j)])
                    end.append(patch_id_5x[str(i - 1) + '-' + str(j + 1)])
               if str(i + 1) + '-' + str(j - 1) in patch_id_5x:
                    start.append(patch_id_5x[str(i) + '-' + str(j)])
                    end.append(patch_id_5x[str(i + 1) + '-' + str(j - 1)])
               if str(i - 1) + '-' + str(j - 1) in patch_id_5x:
                    start.append(patch_id_5x[str(i) + '-' + str(j)])
                    end.append(patch_id_5x[str(i - 1) + '-' + str(j - 1)])

     # 10x resolution edge between nodes
     for x in patch_id:
          if len(x.split('-')) == 2 and x != '-1':
               #             print(x)
               i = int(x.split('-')[0])
               j = int(x.split('-')[1])
               ##add edge with 10x node to its neighbour
               if str(i) + '-' + str(j + 1) in patch_id:
                    start.append(patch_id[str(i) + '-' + str(j)])
                    end.append(patch_id[str(i) + '-' + str(j + 1)])
               if str(i) + '-' + str(j - 1) in patch_id:
                    start.append(patch_id[str(i) + '-' + str(j)])
                    end.append(patch_id[str(i) + '-' + str(j - 1)])
               if str(i + 1) + '-' + str(j) in patch_id:
                    start.append(patch_id[str(i) + '-' + str(j)])
                    end.append(patch_id[str(i + 1) + '-' + str(j)])
               if str(i - 1) + '-' + str(j) in patch_id:
                    start.append(patch_id[str(i) + '-' + str(j)])
                    end.append(patch_id[str(i - 1) + '-' + str(j)])
               if str(i + 1) + '-' + str(j + 1) in patch_id:
                    start.append(patch_id[str(i) + '-' + str(j)])
                    end.append(patch_id[str(i + 1) + '-' + str(j + 1)])
               if str(i - 1) + '-' + str(j + 1) in patch_id:
                    start.append(patch_id[str(i) + '-' + str(j)])
                    end.append(patch_id[str(i - 1) + '-' + str(j + 1)])
               if str(i + 1) + '-' + str(j - 1) in patch_id:
                    start.append(patch_id[str(i) + '-' + str(j)])
                    end.append(patch_id[str(i + 1) + '-' + str(j - 1)])
               if str(i - 1) + '-' + str(j - 1) in patch_id:
                    start.append(patch_id[str(i) + '-' + str(j)])
                    end.append(patch_id[str(i - 1) + '-' + str(j - 1)])

               #     print(start)
     #     print(end)
     return [start, end]


example_edge = get_edge_index_2('normal_001')[0][:20],get_edge_index_2('normal_001')[1][:20]

all_node_type = {}
for x in patients:
     t = []
     for z in t_all_feature[x]:
          if z == '-1.jpeg':
               t.append(0)
     for z in t_all_feature[x]:
          if len(z.split('-')) == 6:
               t.append(1)
     for z in t_all_feature[x]:
          if len(z.split('-')) == 2 and z != '-1.jpeg':
               t.append(2)

     all_node_type[x] = t

example_node = all_node_type['normal_001']

tree = {}
for p_id in patients:
     # t_tree contains the father node idex for each node
     t_tree = np.array([-2] * len(all_node_type[p_id]))
     t_tree[np.where(np.array(all_node_type[p_id]) == 0)] = -1
     t_tree[np.where(np.array(all_node_type[p_id]) == 1)] = np.where(np.array(all_node_type[p_id]) == 0)[0][0]
     #     print(t_tree)

     patch_id = {}
     i = 0
     for x in t_all_feature[p_id]:
          patch_id[x.split('.')[0]] = i
          i += 1
     #     print(patch_id)

     for x in patch_id:
          if len(x.split('-')) == 6:
               if x.split('-')[0] + '-' + x.split('-')[2] in patch_id:
                    t_tree[patch_id[x.split('-')[0] + '-' + x.split('-')[2]]] = patch_id[x]
               if x.split('-')[1] + '-' + x.split('-')[2] in patch_id:
                    t_tree[patch_id[x.split('-')[1] + '-' + x.split('-')[2]]] = patch_id[x]
               if x.split('-')[0] + '-' + x.split('-')[3] in patch_id:
                    t_tree[patch_id[x.split('-')[0] + '-' + x.split('-')[3]]] = patch_id[x]
               if x.split('-')[1] + '-' + x.split('-')[3] in patch_id:
                    t_tree[patch_id[x.split('-')[1] + '-' + x.split('-')[3]]] = patch_id[x]

     tree[p_id] = t_tree
#     break

# node coordinate
x_y = {}
for x in patients:
     dd = []
     for z in t_all_feature[x]:
          if z == '-1.jpeg':
               dd.append((0, 0))

     for z in t_all_feature[x]:
          if len(z.split('-')) == 6:
               dd.append((float(z.split('-')[-2]), float(z.split('-')[-1].split('.')[0])))

     for z in t_all_feature[x]:
          if len(z.split('-')) == 2 and z != '-1.jpeg':
               dd.append((float(z.split('-')[0]), float(z.split('-')[1].split('.')[0])))

     x_y[x] = dd

max_x_y_5x = {}
max_x_y_10x = {}
count =0
for p_id in patients:
     try:
          x = []
          y = []
          for z in np.array(x_y[p_id])[np.where(np.array(all_node_type[p_id]) == 1)]:
               x.append(z[0])
               y.append(z[1])
          max_x_y_5x[p_id] = [np.max(np.array(x)), np.max(np.array(y))]

          x = []
          y = []
          for z in np.array(x_y[p_id])[np.where(np.array(all_node_type[p_id]) == 2)]:
               x.append(z[0])
               y.append(z[1])
          max_x_y_10x[p_id] = [np.max(np.array(x)), np.max(np.array(y))]
     except:
          print("failed for id: " + str(p_id))
          del copy_patient_label[p_id]
          count+=1

x_y = {}
for x in patients:
     dd = []
     for z in t_all_feature[x]:
          if z == '-1.jpeg':
               dd.append((0, 0))

     for z in t_all_feature[x]:
          if len(z.split('-')) == 6:
               dd.append((float(z.split('-')[-2]) / max_x_y_5x[x][0],
                          float(z.split('-')[-1].split('.')[0]) / max_x_y_5x[x][1]))

     for z in t_all_feature[x]:
          if len(z.split('-')) == 2 and z != '-1.jpeg':
               dd.append((float(z.split('-')[0]) / max_x_y_10x[x][0],
                          float(z.split('-')[1].split('.')[0]) / max_x_y_10x[x][1]))

     x_y[x] = dd


all_data = {}
for id in patients:
#     print(id)
    node_attr=torch.tensor(all_feature[id],dtype=torch.float)
    edge_index_tree_8nb = torch.tensor(get_edge_index_2(id),dtype=torch.long)
    batch = torch.tensor([0 for i in range(len(node_attr))])
    node_type = torch.tensor(all_node_type[id])
    node_tree = torch.tensor(tree[id])
    x_y_index = torch.tensor(x_y[id])
    data_id = id
    data = Data(x=node_attr,edge_index_tree_8nb=edge_index_tree_8nb,data_id=data_id,batch =batch,node_type=node_type,node_tree=node_tree,x_y_index=x_y_index)
    all_data[id] = data
#     print(data)


joblib.dump(all_data, save_path)

print("a")

import pickle
with open('WSI_processing/real_ESCA_patient_label.pkl', 'wb') as file:
    pickle.dump(copy_patient_label, file)

print(count)