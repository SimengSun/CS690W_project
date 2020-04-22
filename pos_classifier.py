import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class ConvClassifier(nn.Module):

	def __init__(self, conv1_out=3, conv1_k=3, pool1=5,
					   conv2_out=3, conv2_k=3, pool2=5,
					   fc_dim=256, k=9, fc1_dim=120
				):

		super(ConvClassifier, self).__init__()

		self.conv1 = torch.nn.Conv1d(1, conv1_out, conv1_k)
		self.pool1 = nn.MaxPool1d(pool1)
		self.conv2 = torch.nn.Conv1d(conv1_out, conv2_out, conv2_k)
		self.pool2 = nn.MaxPool1d(pool2)
		self.conv3 = torch.nn.Conv1d(conv2_out, conv2_out, conv2_k)
		self.pool3 = nn.MaxPool1d(pool2)

		self.fc1 = nn.Linear(fc1_dim, fc_dim)
		self.fc2 = nn.Linear(fc_dim, k)

		self.reset_parameters()

	def reset_parameters(self):
		gain = nn.init.calculate_gain('linear')
		nn.init.xavier_uniform_(self.fc1.weight, gain)
		nn.init.xavier_uniform_(self.fc2.weight, gain)

	def forward(self, x):

		n = x.shape[0]

		x = x[:, None, :]
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = self.pool3(F.relu(self.conv3(x)))
		
		x = x.view(n, -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return x

class SimpleClassifier(nn.Module):

	def __init__(self, f_dim=5096, fc1_dim=512, fc2_dim=100, k=9
				):

		super(SimpleClassifier, self).__init__()

		self.fc1 = nn.Linear(f_dim, fc1_dim)
		self.fc2 = nn.Linear(fc1_dim, fc2_dim)
		self.fc = nn.Linear(fc2_dim, k)

		self.reset_parameters()

	def reset_parameters(self):
		gain = nn.init.calculate_gain('linear')
		nn.init.xavier_uniform_(self.fc1.weight, gain)
		nn.init.xavier_uniform_(self.fc2.weight, gain)
		nn.init.xavier_uniform_(self.fc.weight, gain)

	def forward(self, x):

		n = x.shape[0]
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc(x)

		return x