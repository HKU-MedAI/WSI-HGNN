from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
import joblib as joblib

plt.ion()   # interactive mode
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 80
num_workers = 16
KimiaNetPyTorchWeights_path = ""
image_type = 'jpeg'
'''
dataset_folder like:
-dataset_folder
	-class_1
		-wsi_1
			-1_1.jpeg
			...
		-wsi_2
		...
		-wsi_k
	-class_2
	...
	-class_k
'''
dataset_folder = glob('/*')
'''
output file like:
{
	'wsi_1': {
		'1_1.jpeg': 1024 tensor
		...
	}
	...
	'wsi_n': {...}
}
'''
save_path = ''

trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# trans = transforms.Compose([transforms.ToTensor()])

data_transforms = {
	'train': transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}


class My_dataloader(Dataset):
	def __init__(self, data_24, transform):
		"""
		Args:
			data_24: path to input data
		"""
		self.data_24 = data_24
		self.pathes_24 = glob(self.data_24+'/*.'+image_type)
		self.transform = transform

	def __len__(self):
		return len(self.pathes_24)

	def __getitem__(self, idx):
		img_24 = Image.open(self.pathes_24[idx]).convert('RGB')
		img_24 = img_24.resize((512, 512), Image.ANTIALIAS)
		img_24_name = self.pathes_24[idx].split('/')[-1]
		img_24_folder = self.pathes_24[idx].split('/')[-2]
		if self.transform:
			img_24 = self.transform(img_24)
		return img_24, img_24_name, img_24_folder


def test_model(model, criterion, num_epochs=25):
	since = time.time()
	model.eval()   # Set model to evaluate mode

	wsi_counter = 0
	no_sort_t_all_feature = dict()
	for class_idx in range(len(dataset_folder)):
		class_folder = dataset_folder[class_idx]
		wsi_folder = glob(class_folder + '/*')
		for wsi_index in range(len(wsi_folder)):
			slide_patches_dict_1024 = {}
			test_path_test = wsi_folder[wsi_index]
			test_imagedataset = My_dataloader(test_path_test, trans)
			dataloader_test = torch.utils.data.DataLoader(test_imagedataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
			# Iterate over data.
			for ii, (inputs, img_name, folder_name) in enumerate(dataloader_test):
				inputs = inputs.to(device)
				output1, outputs = model(inputs)
				print(output1.shape)
				output_1024 = output1.cpu().detach().numpy()
				for j in range(len(outputs)):
					slide_patches_dict_1024[img_name[j]] = output_1024[j]
			no_sort_t_all_feature[test_path_test.split('/')[-1]] = slide_patches_dict_1024
			time_elapsed = time.time() - since
			print('Evaluation completed in {:.0f}m {:.0f}s for wsi {:d}'.format(time_elapsed // 60, time_elapsed % 60, wsi_counter))
			wsi_counter = wsi_counter + 1

	return no_sort_t_all_feature


class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		out_3 = self.fc_4(x)
		return  out_1, out_3


model = torchvision.models.densenet121(pretrained=True)
for param in model.parameters():
	param.requires_grad = False
model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
num_ftrs = model.classifier.in_features
model_final = fully_connected(model.features, num_ftrs, 30)
model = model.to(device)
model_final = model_final.to(device)
model_final = nn.DataParallel(model_final)
params_to_update = []
criterion = nn.CrossEntropyLoss()

state_dict = torch.load(KimiaNetPyTorchWeights_path)
model_final.load_state_dict(state_dict)

no_sort_t_all_feature = test_model(model_final, criterion, num_epochs=1)
joblib.dump(no_sort_t_all_feature, save_path)
