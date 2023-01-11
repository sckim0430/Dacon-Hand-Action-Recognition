import pickle

with open('./tools/data/skeleton/ntu120_xsub_train.pkl', 'rb') as f:
	data = pickle.load(f)

print(data)