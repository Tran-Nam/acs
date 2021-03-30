import os
import numpy as np 
from dataset import Dataset
from optimizer import ACS
from tqdm import tqdm

data_dir = '../data/wtsds-instances'
paths = os.listdir(data_dir)
paths = [path for path in paths if path.endswith('instance')]

out_dir = 'out'
os.makedirs(out_dir, exist_ok=True)

optimizer = ACS(ants=10)

for path in tqdm(paths):
    data_obj = Dataset(os.path.join(data_dir, path))
    loss = optimizer.fit(data_obj, iterations=100)
    np.save(os.path.join(out_dir, path.split('.')[0] + '.npy'), loss)

# data_obj = Dataset('../data/wtsds-instances/wt_sds_3.instance')
# optimizer = ACS(ants=10)

# dummy
# skip = 100
# data_obj.ps = data_obj.ps[:skip]
# data_obj.ds = data_obj.ds[:skip]
# # data_obj.ds = np.array([int(i/10) for i in data_obj.ds])
# data_obj.ws = data_obj.ws[:skip]
# data_obj.setup_time = data_obj.setup_time[:skip, :skip]
# # data_obj.setup_time = np.ones((skip, skip))*100


# print('process', data_obj.ps)
# print('due date', data_obj.ds)
# print('weight', data_obj.ws)
# print('setup', data_obj.setup_time)

# optimizer.data_obj = data_obj 
# solution = range(skip)
# coord, path, C = optimizer._evaluate([solution])
# print('coord', coord)
# print('path', path)
optimizer.fit(data_obj, iterations=100)
# print(C)