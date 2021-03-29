import os
import numpy as np 
from dataset import Dataset
from optimizer import ACS

data_obj = Dataset('../data/wtsds-instances/wt_sds_1.instance')
optimizer = ACS(ants=10)

# dummy
skip = 5
data_obj.ps = data_obj.ps[:skip]
data_obj.ds = data_obj.ds[:skip]
data_obj.ds = np.array([int(i/10) for i in data_obj.ds])
data_obj.ws = data_obj.ws[:skip]
# data_obj.setup_time = data_obj.setup_time[:skip, :skip]
data_obj.setup_time = np.ones((skip, skip))*100


print('process', data_obj.ps)
print('due date', data_obj.ds)
print('weight', data_obj.ws)
print('setup', data_obj.setup_time)

optimizer.data_obj = data_obj 
solution = range(skip)
# coord, path, C = optimizer._evaluate([solution])
# print('coord', coord)
# print('path', path)
optimizer.fit(data_obj, iterations=1000)
# print(C)