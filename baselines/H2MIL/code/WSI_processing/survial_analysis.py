import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def function(a, b):
    if a == b:
        return 1
    else:
        return 0

clinical_data = pd.read_csv('sarc_tcga_pan_can_atlas_2018_clinical_data.tsv', sep='\t')

# print(pd.isnull(clinical_data).any())
# print(clinical_data.describe())
#
# analysis_frequency = clinical_data[['Cancer Type Detailed',
#                           'MSIsensor Score'
#                          ]]
# analysis_frequency['Cancer Type Detailed'] = analysis_frequency['Cancer Type Detailed'].replace(
#     'Dedifferentiated Liposarcoma', 'DDLS')
# analysis_frequency['Cancer Type Detailed'] = analysis_frequency['Cancer Type Detailed'].replace(
#     'Leiomyosarcoma', 'LMS')
# analysis_frequency['Cancer Type Detailed'] = analysis_frequency['Cancer Type Detailed'].replace(
#     'Malignant Peripheral Nerve Sheath Tumor', 'MPNST')
# analysis_frequency['Cancer Type Detailed'] = analysis_frequency['Cancer Type Detailed'].replace(
#     'Myxofibrosarcoma', 'MFS')
# analysis_frequency['Cancer Type Detailed'] = analysis_frequency['Cancer Type Detailed'].replace(
#     'Synovial Sarcoma', 'SYNS')
# analysis_frequency['Cancer Type Detailed'] = analysis_frequency['Cancer Type Detailed'].replace(
#     'Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma', 'UPS')
# analysis_frequency['Cancer Type Detailed'] = analysis_frequency['Cancer Type Detailed'].replace(
#     'Desmoid/Aggressive Fibromatosis', 'DROP')
#
# # draw a scatter plot of x vs y.
# plt.scatter(analysis_frequency.iloc[:,0],analysis_frequency.iloc[:,1])
# plt.xlabel('Cancer')
# plt.ylabel('MSIsensor Score')
# plt.show()
#
# data = pd.read_table('combined_pancancer.txt' ,header=None, encoding='gb2312', sep='\t')

survival = clinical_data[['Patient ID',
                          'Months of disease-specific survival',
                          'Disease-specific Survival status',
                          'Overall Survival (Months)',
                          'Overall Survival Status',
                          'Progress Free Survival (Months)',
                          'Progression Free Status',
                          'Diagnosis Age']]






df = survival
df['bool'] = df.apply(lambda x : function(x['Months of disease-specific survival'],x['Overall Survival (Months)']),axis = 1)

patient_list = survival['Patient ID'].values.tolist()
survival_months_list = survival['Overall Survival (Months)'].values.tolist()
survival_status_list = survival['Overall Survival Status'].values.tolist()
age_list = survival['Diagnosis Age'].values.tolist()

survival_months_np = np.array((survival_months_list))
quantile_25 = np.percentile(survival_months_np, 25)
quantile_50 = np.percentile(survival_months_np, 50)
quantile_75 = np.percentile(survival_months_np, 75)

survival_data = dict()
count = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}
for i in range(len(patient_list)):
    survival_data[patient_list[i]] = dict()

    c = survival_status_list[i][0]
    if c == '0':
        survival_data[patient_list[i]]['c'] = 1
    elif c == '1':
        survival_data[patient_list[i]]['c'] = 0

    time = survival_months_list[i]
    survival_data[patient_list[i]]['event_time'] = time
    if time < quantile_25:
        survival_data[patient_list[i]]['survival_discrete_label'] = 0
    elif quantile_25 <= time and time < quantile_50:
        survival_data[patient_list[i]]['survival_discrete_label'] = 1
    elif quantile_50 <= time and time < quantile_75:
        survival_data[patient_list[i]]['survival_discrete_label'] = 2
    elif quantile_75 <= time:
        survival_data[patient_list[i]]['survival_discrete_label'] = 3

    count[survival_data[patient_list[i]]['survival_discrete_label']] += 1

    survival_data[patient_list[i]]['diagnosis_age'] = age_list[i]

joblib.dump(survival_data,'/home/r10user3/Documents/H2_workstation10/code/WSI_processing/patient_survival_data.pkl')
print("a")