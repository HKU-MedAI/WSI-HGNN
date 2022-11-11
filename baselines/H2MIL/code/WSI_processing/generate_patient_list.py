import joblib as joblib
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
filename = "/home/r10user3/Documents/H2_workstation10/code/WSI_processing/combined_clinical.tsv"
sarc_data = pd.read_csv(filename, sep='\t')
patient_and_cancer = sarc_data[['File.Name', 'Oncotree.Code', 'Tumor.Disease.Anatomic.Site']]
patient_and_cancer_list = list(patient_and_cancer.values)
patient_and_cancer_new = list()
for x in patient_and_cancer_list:
    subtype = x[1]
    subtype = subtype.replace(' ', "_")
    if '/' in subtype:
        subtype = subtype.split('/')[0].split('.')[0]

    if subtype == 'LMS':
        if x[2]=='Uterus' or x[2]=='Pelvis|Uterus' or x[2]=='Cervix|Uterus':
            continue

    if subtype == 'DES':
        continue

    patient_and_cancer_new.append([x[0].split('.')[0][0:-7], subtype, x[2]])

subtype_label = {
    'DDLS': 0,
    'LMS': 1,
    'MFS': 2,
    'MPNST': 3,
    'SYNS': 4,
    'UPS': 5
}

no_rep_patient_and_cancer_new = dict()
select_list = list()
for x in patient_and_cancer_new:
    if x[0] not in select_list:
        select_list.append(x[0])
        no_rep_patient_and_cancer_new[x[0]] = subtype_label[x[1]]

    elif no_rep_patient_and_cancer_new[x[0]] != subtype_label[x[1]]:
        print("shit")

for i in range(6):
    count = 0
    for x in no_rep_patient_and_cancer_new:
        if no_rep_patient_and_cancer_new[x] == i:
            count = count + 1
    print(str(i)+" "+str(count))

joblib.dump(no_rep_patient_and_cancer_new,'/home/r10user3/Documents/H2_workstation10/code/WSI_processing/patient_wise_and_label.pkl')