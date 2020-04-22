
import os
import random
import pickle
import numpy as np 

random.seed(42)

def load_data(fn):
	return np.loadtxt(fn)

POS = ['x1_y1', 'x1_y2', 'x1_y3', 
       'x2_y1', 'x2_y2', 'x2_y3', 
       'x3_y1', 'x3_y2', 'x3_y3']

def get_pos_data(idx):
    p = POS[idx]
    fpath_prefix = f"../Segment/{p}/"
    fnames = [f for f in os.listdir(fpath_prefix) if f.startswith(p)]

    ds = []
    for fi in range(1, len(fnames)+1):
        d = load_data(f"{fpath_prefix}{p}_{fi}.csv")
        ds.append(d)
    ds = np.stack(ds)

    n = ds.shape[0]
    train_idx = random.sample(range(n), int(n*0.8))
    test_idx = [i for i in range(n) if i not in train_idx]

    ds_train = ds[train_idx]
    ds_test = ds[test_idx]
    y_train = np.ones(ds_train.shape[0]) * idx
    y_test = np.ones(ds_test.shape[0]) * idx
    return ds_train, y_train, ds_test, y_test

x_train_all = []
y_train_all = []
x_test_all = []
y_test_all = []

for i in range(len(POS)):
	x_train, y_train, x_test, y_test = get_pos_data(i)
	x_train_all.append(x_train)
	x_test_all.append(x_test)
	y_train_all.append(y_train)
	y_test_all.append(y_test)

x_train_all = np.concatenate(x_train_all, axis=0)
y_train_all = np.concatenate(y_train_all)
x_test_all = np.concatenate(x_test_all, axis=0)
y_test_all = np.concatenate(y_test_all)

with open("../data.pkl", "wb") as f:
	pickle.dump({
		"x_train": x_train_all, 
		"y_train": y_train_all,
		"x_test": x_test_all,
		"y_test": y_test_all
		}, f, protocol=pickle.HIGHEST_PROTOCOL)

