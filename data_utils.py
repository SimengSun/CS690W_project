
import torch
import pdb
import math
import random
import numpy as np
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
random.seed(42)

def get_dataloader(data, split, num_workers=2, bsz=50):

	x, y = data[f'x_{split}'], data[f'y_{split}']
	dataset = Dataset(x, y)
	dataloader = DataLoader(dataset, num_workers=2, batch_size=bsz)

	return dataloader

class Dataset(torch.utils.data.IterableDataset):

	def __init__(self, x, y):

		self.x = x
		self.y = y
		shuffle_idx = random.sample(range(x.shape[0]), x.shape[0])
		self.x = self.x[shuffle_idx]
		self.y = self.y[shuffle_idx]
		self.data = torch.tensor(np.concatenate((self.x, self.y[:, np.newaxis]), axis=1)).float()

		self.start = 0
		self.end = self.data.size(0)

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None: 
			iter_start = self.start
			iter_end = self.end
		else:  					
			per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
			worker_id = worker_info.id
			iter_start = self.start + worker_id * per_worker
			iter_end = min(iter_start + per_worker, self.end)
			
		return iter(self.data[iter_start:iter_end])