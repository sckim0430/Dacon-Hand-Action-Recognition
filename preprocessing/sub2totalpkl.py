import os
import pickle

train_path = '/home/sckim/Dataset/Competition/dacon_hand/train_pkl'
test_path = '/home/sckim/Dataset/Competition/dacon_hand/test_pkl'

datas = []

for tf in os.listdir(train_path):
    with open(os.path.join(train_path, tf), 'rb') as f:
        data = pickle.load(f)
        f.close()
    datas.append(data)

with open(os.path.join('/home/sckim/Dataset/Competition/dacon_hand/train.pkl'), 'wb') as f:
    pickle.dump(datas, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

datas_2 = []
for tf in os.listdir(test_path):
    with open(os.path.join(test_path, tf), 'rb') as f:
        data = pickle.load(f)
        f.close()
    datas_2.append(data)

with open(os.path.join('/home/sckim/Dataset/Competition/dacon_hand/test.pkl'), 'wb') as f:
    pickle.dump(datas_2, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
