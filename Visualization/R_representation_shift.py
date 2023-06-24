import sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer')
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from scipy.stats import wasserstein_distance
import numpy as np
from Datasets.create_dataset import Dataset_wrap, SkinDataset, norm01
import pandas as pd

#dataset
# path_to_downloaded_data= 'CIFAR10_data'
# trans = transforms.Compose([
#     transforms.ToTensor()
# ])


# cifar_data = datasets.CIFAR10(root=path_to_downloaded_data, train=True, download=True, transform=trans)

# in_distribution_dataset = torch.utils.data.DataLoader(cifar_data, batch_size=1)


# ood dataset
# 1. CIFAR10-C:
# cifar_corr_data = TensorDataset(torch.from_numpy(np.load('CIFAR10-C_data/CIFAR-10-C/contrast.npy')))
train_loaders = {}  # initialize data loaders
val_loaders = {}
test_loaders = {}
dataset_list = ['isic2018','PH2','DMF','SKD']
for dataset_name in dataset_list:
	datas = Dataset_wrap(use_old_split=True, img_size=256, dataset_name = dataset_name, split_ratio=[0.6,0.2,0.2], 
	train_aug=False, data_folder='/bigdata/siyiplace/data/skin_lesion')
	train_data, val_data, test_data = datas['train'], datas['val'], datas['test']
	train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_data, val_data]),
											batch_size=8,
											shuffle=False,
											num_workers=0,
											pin_memory=True,
											drop_last=True)
	val_loader = torch.utils.data.DataLoader(val_data,
											batch_size=8,
											shuffle=False,
											num_workers=0,
											pin_memory=True,
											drop_last=False)
	test_loader = torch.utils.data.DataLoader(test_data,
											batch_size=8,
											shuffle=False,
											num_workers=0,
											pin_memory=True,
											drop_last=False)
	train_loaders[dataset_name] = train_loader
	val_loaders[dataset_name] = val_loader
	test_loaders[dataset_name] = test_loader



########################################################################################################
# network 
class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
	
	def forward(self, x):
		return x

# model = models.resnet(pretrained=True)
model = models.resnet18(pretrained=True)
setattr(model, 'fc', Identity())
model.cuda()

# Download test data from open datasets.
# trans = transforms.Compose([
# 				transforms.Resize(128),
# 				transforms.ToTensor()
# ])

# calculate Representation Shift: 
# 1. Extract activations
def extract_activations(model, dataloader, max_samples=1000):
	"""
	Iterate though each (subset of) dataset and store activations.
	
	Parameters:
					model (torch.model): the model to evaluate, output representation of input image with size D
					dataloader (torch.utils.data.DataLoader): Dataloader, with batch size 1
					max_samples (int): number of samples to evaluate, N

	Returns:
					activations (numpy.array): Array of size NxD
	"""
	model.eval()

	activations = []
	with torch.no_grad():
		for idx, batch in enumerate(dataloader):
			img = batch['image'].cuda().float()
			if idx >= max_samples:
				break
			print(f'\r{idx}/{min(len(dataloader), max_samples)}', end="")
			out = model(img)
			# out = model(batch[0])
			activations.extend(out.cpu().numpy())
	
	return np.asarray(activations)

# 2. Measure R using Wasserstein distance 
def representation_shift(act_ref, act_test):
	"""
	Calculate representation shift using Wasserstein distance
	
	Parameters:
					act_ref (numpy.array): Array of size NxD
					act_test (numpy.array): Array of size NxD

	Returns:
					representation_shift (float): Mean Wasserstein distance over all channels (D) 
	"""
	wass_dist = [wasserstein_distance(act_ref[:, channel], act_test[:, channel]) for channel in range(act_ref.shape[1])]
	return np.asarray(wass_dist).mean()


# wass_dist_array = np.zeros((4,7))
wass_dist_array = np.zeros((4,5))
for i in range(len(dataset_list)):  # in domain dataset index
	for j in range(len(dataset_list)):  # out domain dataset index
		# Get activations for a subset of each dataset
		activations_training_data = extract_activations(model, train_loaders[dataset_list[i]])
		# activations_val_data_indistribution = extract_activations(model, val_loaders[dataset_list[i]])
		activations_test_data_indistribution = extract_activations(model, test_loaders[dataset_list[i]])
		activations_test_data_OOD = extract_activations(model, test_loaders[dataset_list[j]])

		# compute R
		# wass_dist_indist_TR2V = representation_shift(activations_training_data, activations_val_data_indistribution)
		wass_dist_indist_TR2T = representation_shift(activations_training_data, activations_test_data_indistribution)
		# wass_dist_indist_V2T = representation_shift(activations_test_data_indistribution, activations_val_data_indistribution)
		wass_dist_outdist = representation_shift(activations_training_data, activations_test_data_OOD)

		# wass_dist_array[i,:3] = [wass_dist_indist_TR2V,wass_dist_indist_TR2T,wass_dist_indist_V2T]
		# wass_dist_array[i,j+3] = wass_dist_outdist
		wass_dist_array[i,:1] = [wass_dist_indist_TR2T]
		wass_dist_array[i,j+1] = wass_dist_outdist
        
		print('in domain {}, out domain {}'.format(dataset_list[i], dataset_list[j]))
		# print('Representation shift, in-distribution Train 2 Val:', wass_dist_indist_TR2V)
		print('Representation shift, in-distribution Train 2 Test:', wass_dist_indist_TR2T)
		# print('Representation shift, in-distribution Test 2 Val:', wass_dist_indist_V2T)
		#print('imbalanced (dataset as it is)') 
		# print('balanced training and test dataset') 
		print('Representation shift, out-of-distribution:', wass_dist_outdist) 
		print('\n')
	# 	break
	# break
	


print(wass_dist_array)
# headers = ['TR2V','TR2T','V2T','ISIC','PH2','DMF','SKD']
headers = ['TR2T','ISIC','PH2','DMF','SKD']
df = pd.DataFrame(np.round(wass_dist_array,5), columns=headers)
df.to_csv('R_Values.csv', index=False)
# Get activations for a subset of each dataset
# activations_training_data = extract_activations(model, HAM_training_dataset)
# activations_test_data_indistribution = extract_activations(model, HAM_test_dataset)
# activations_test_data_OOD = extract_activations(model, OOD_test_dataset)

# Compute R
# wass_dist_indist = representation_shift(activations_training_data, activations_test_data_indistribution)
# wass_dist_outdist =  representation_shift(activations_training_data, activations_test_data_OOD)

# print('Representation shift, in-distribution:', wass_dist_indist)
# #print('imbalanced (dataset as it is)') 
# print('balanced training and test dataset') 
# print('Representation shift, out-of-distribution:', wass_dist_outdist) 




