"""Merge the Open Dataset(NTU) and Custom Dataset consist of Train / Validation / Test 
"""
import random
import math
import numpy as np
import os
import pickle

custom_train_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/custom_train.pkl'
custom_val_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/custom_val.pkl'
custom_test_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/custom_test.pkl'

train_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/total_set_pkl/total_train.pkl'
val_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/total_set_pkl/total_val.pkl'
test_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/total_set_pkl/total_test.pkl'

ntu_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/ntu120_xsub_train.pkl'

# Load Open Dataset and Custom Dataset PKL File
with open(custom_train_path, 'rb') as f:
    custom_train_data = pickle.load(f)
f.close()

with open(custom_val_path, 'rb') as f:
    custom_val_data = pickle.load(f)
f.close()

with open(custom_test_path, 'rb') as f:
    custom_test_data = pickle.load(f)
f.close()

with open(train_path, 'rb') as f:
    train_data = pickle.load(f)
f.close()

with open(val_path, 'rb') as f:
    val_data = pickle.load(f)
f.close()

with open(test_path, 'rb') as f:
    test_data = pickle.load(f)
f.close()

# Ntu PKL File which have Not Falling Label will be append to Total Dataset PKL File
with open(ntu_path, 'rb') as f:
    ntu_data = pickle.load(f)
f.close()

data = []

for d in custom_train_data:
    if d["label"]:
        data.append(d)

for d in custom_val_data:
    if d["label"]:
        data.append(d)

for d in custom_test_data:
    if d["label"]:
        data.append(d)

#Get Num of Not Falling Label to Match with Num of Falling Label in (Custom Dataset + Open Dataset)
total_neg_data_num = 0
neg_data_num = 0

for d in train_data:
    if not d["label"]:
        total_neg_data_num += 1

for d in val_data:
    if not d["label"]:
        total_neg_data_num += 1

for d in test_data:
    if not d["label"]:
        total_neg_data_num += 1

# To avoid class imbalance problem, So We Calculate Each Label Num with Divide Num of Falling Label to 119
len_positive = len(data)
each_num = math.ceil(len_positive/119.0)

num_array = np.zeros(120)
data_neg = []

# Append the Not Falling Label Data
for idx, ntu in enumerate(ntu_data):
    if ntu["label"] == 42:
        continue

    if neg_data_num <= total_neg_data_num and num_array[ntu["label"]] >= each_num:
        neg_data_num += 1
        continue

    num_array[ntu["label"]] += 1
    ntu["label"] = 0
    ntu["frame_dir"] = ntu["frame_dir"].split('A')[0]+'A001'
    data_neg.append(ntu)

    if np.sum(num_array) >= len_positive:
        break

# Shuffle Data
for _ in range(10):
    random.shuffle(data)
    random.shuffle(data_neg)

#Train / Validation / Test Ratio = 0.85 / 0.1 / 0.05
train_ratio = 0.85
val_ratio = 0.1

train_pos_len = int(train_ratio * len_positive)
val_pos_len = int(val_ratio * len_positive)
test_pos_len = len_positive - train_pos_len - val_pos_len

total_train = train_data
total_val = val_data
total_test = test_data

for idx in range(train_pos_len):
    total_train.append(data[idx])
    total_train.append(data_neg[idx])

for idx in range(val_pos_len):
    total_val.append(data[train_pos_len + idx])
    total_val.append(data_neg[train_pos_len + idx])

for idx in range(test_pos_len):
    total_test.append(data[train_pos_len+val_pos_len+idx])
    total_test.append(data_neg[train_pos_len+val_pos_len+idx])

for _ in range(10):
    random.shuffle(total_train)
    random.shuffle(total_val)
    random.shuffle(total_test)

#Generate Train / Validation / Test Dataset PKL File
train_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/total_set_pkl/total_train.pkl'
val_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/total_set_pkl/total_val.pkl'
test_path = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/total_set_pkl/total_test.pkl'

with open(train_path, "wb") as f:
    pickle.dump(total_train, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

with open(val_path, "wb") as f:
    pickle.dump(total_val, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

with open(test_path, "wb") as f:
    pickle.dump(total_test, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

print("done")
