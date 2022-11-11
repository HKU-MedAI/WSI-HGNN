import shutil

import os

import json
import pandas as pd

def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

'''
pair the patient and their cancer subtype
'''
filename = "/home/r10user3/Documents/TCGA/SARC/combined_clinical.tsv"
sarc_data = pd.read_csv(filename, sep='\t')
patient_and_cancer = sarc_data[['File.Name', 'Oncotree.Code']]
patient_and_cancer_list = list(patient_and_cancer.values)
patient_and_cancer_new = list()
for x in patient_and_cancer_list:
    subtype = x[1]
    subtype = subtype.replace(' ', "_")
    if '/' in subtype:
        subtype = subtype.split('/')[0]
    patient_and_cancer_new.append([x[0], subtype])

file_path = "/home/r10user3/Documents/TCGA/SARC"
list_name = list()
listdir(file_path, list_name)

svs_list = list()
for file in list_name:
    if file.endswith(".svs"):
        svs_list.append(file)

source2des = list()
for svs in svs_list:
    for record in patient_and_cancer_new:
        flag = 0
        if record[0] in svs:
            # print(record[0] + " " + svs)
            filename = svs.split('/')[-1]
            source2des.append([svs, "/home/r10user3/Documents/TCGA/"+record[1]+'/'+filename])

for x in source2des:
    shutil.move(x[0], x[1])
