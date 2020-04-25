import os
import torch
import pickle
import argparse
import numpy as np
from torch import nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from pos_classifier import ConvClassifier, Conv2dClassifier, SimpleClassifier
from data_utils import Dataset, get_dataloader


def main(args):


	# load data
	with open(args.data, "rb") as f:
		data = pickle.load(f)

	# get dataloader
	train_dataloader = get_dataloader(data, 'train', bsz=args.bsz, freq_dom=args.freq_dom)
	val_dataloader = get_dataloader(data, 'test', bsz=200, freq_dom=args.freq_dom) 
	
	# build model
	if args.mdl == 'maxent':
		model = SimpleClassifier() 

	elif args.mdl == 'conv':
		model = ConvClassifier()

	elif args.mdl == 'conv2d':
		model = Conv2dClassifier()

	if args.gpu:
		model = model.cuda()

	crit = nn.CrossEntropyLoss()
	# optimizer = optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-9)
	optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

	# train
	losses = 0
	step = 0
	val_acc = []
	for epoch in range(args.max_epoch):
		for i, batch in enumerate(iter(train_dataloader)):
			
			x, y = batch[:, :-1], batch[:, -1].long()

			if args.mdl == "conv2d":
				x = x.view(-1, 129, 158)

			optimizer.zero_grad()
			out = model(x)
			loss = crit(out, y)
			loss.backward()
			optimizer.step()

			losses += loss.item()
			if step % args.log_every == 0:
				print('-epoch {:3} step {:5} train loss {:5}'.format(epoch, i, loss.item()))

			if step % args.eval_every == 0:
				model.eval()

				true_y, pred_y = [], []
				for i, b in  enumerate(val_dataloader):
					x, y = b[:, :-1], b[:, -1].long()
					if args.mdl == "conv2d":
						x = x.view(-1, 129, 158)
					true_y.append(y)
					out = model(x)
					out = out.argmax(dim=1)
					pred_y.append(out)

				true_y, pred_y = torch.cat(true_y), torch.cat(pred_y)
				acc = (true_y == pred_y).sum()/float(true_y.shape[0])
				print('-epoch {:3} step {:5} eval acc {:5}'.format(epoch, i, acc))
				val_acc.append(acc)
				model.train()

				if len(val_acc) > args.early_stop:
					val_acc.pop(0)
				if acc < min(val_acc):
					break
				
			step += 1

	torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()}, os.path.join(args.ckpt_dir, f"model_{args.mdl}.pt"))
	print('acc:', acc)
	print('confusion matrix', confusion_matrix(true_y.tolist(), pred_y.tolist()))

def parse_args():
	parser = argparse.ArgumentParser(description='cs690w time domain position classifier')
	parser.add_argument('--data', type=str, default='../data.pkl')
	parser.add_argument('--freq-dom', default=False, action="store_true")
	parser.add_argument('--mdl', type=str, default='conv2d', choices=['maxent', 'conv', 'conv2d'])
	parser.add_argument('--log-every', type=int, default=1)
	parser.add_argument('--eval-every', type=int, default=50)
	parser.add_argument('--max-epoch', type=int, default=150)
	parser.add_argument('--bsz', type=int, default=200)
	parser.add_argument('--ckpt-dir', type=str, default='./')
	parser.add_argument('--early-stop', type=int, default=3)
	parser.add_argument('--gpu', default=False, action="store_true")
	return parser.parse_args()

if __name__ == "__main__":

	args = parse_args()
	main(args)
