import os
import torch
import pickle
import argparse
import numpy as np
from torch import nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from pos_classifier import ConvClassifier, SimpleClassifier
from data_utils import Dataset, get_dataloader


def main(args):


	# load data
	with open("../data.pkl", "rb") as f:
		data = pickle.load(f)

	# get dataloader
	train_dataloader = get_dataloader(data, 'train', bsz=args.bsz)
	val_dataloader = get_dataloader(data, 'test', bsz=200) 
	
	# build model
	model = SimpleClassifier() if args.mdl == 'maxent' else ConvClassifier()
	crit = nn.CrossEntropyLoss()
	# optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
	optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

	# train
	losses = 0
	step = 0
	val_acc = []
	for epoch in range(args.max_epoch):
		for i, batch in enumerate(iter(train_dataloader)):
			
			x, y = batch[:, :-1], batch[:, -1].long()
			optimizer.zero_grad()
			out = model(x)
			loss = crit(out, y)
			loss.backward()
			optimizer.step()

			losses += loss.item()
			if step % args.eval_every == 0:
				print('-epoch {:3} step {:5} train loss {:5}'.format(epoch, i, loss.item()))

			if step % args.eval_every == 0:
				model.eval()

				true_y, pred_y = [], []
				for i, b in  enumerate(val_dataloader):
					x, y = b[:, :-1], b[:, -1].long()
					true_y.append(y)
					out = model(x)
					out = out.argmax(dim=1)
					pred_y.append(out)

				true_y, pred_y = torch.cat(true_y), torch.cat(pred_y)
				acc = sum(true_y == pred_y)/float(true_y.shape[0])
				print('-epoch {:3} step {:5} eval acc {:5}'.format(epoch, i, acc))
				val_acc.append(acc)
				model.train()

				if len(val_acc) > args.early_stop:
					val_acc.pop(0)
				if acc < min(val_acc):
					break
				
			step += 1

	torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()}, os.path.join(args.ckpt_dir, "model.pt"))
	print('acc:', acc)
	print('confusion matrix', confusion_matrix(true_y.tolist(), pred_y.tolist()))

def parse_args():
	parser = argparse.ArgumentParser(description='cs690w time domain position classifier')
	parser.add_argument('--data', type=str, default='../data.pkl')
	parser.add_argument('--mdl', type=str, default='maxent', choices=['maxent', 'conv'])
	parser.add_argument('--log-every', type=int, default=50)
	parser.add_argument('--eval-every', type=int, default=50)
	parser.add_argument('--max-epoch', type=int, default=150)
	parser.add_argument('--bsz', type=int, default=200)
	parser.add_argument('--ckpt-dir', type=str, default='./')
	parser.add_argument('--early-stop', type=int, default=3)
	return parser.parse_args()

if __name__ == "__main__":

	args = parse_args()
	main(args)
