import os
import numpy as np 
from dataset import Dataset
from optimizer import ACS

data_obj = Dataset('../data/wtsds-instances/wt_sds_1.instance')
optimizer = ACS()

# dummy
data_obj.ps = data_obj.ps[:60]
data_obj.ds = data_obj.ds[:60]
data_obj.ws = data_obj.ws[:60]
data_obj.setup_time = data_obj.setup_time[:60, :60]

"""
print('process', data_obj.ps)
print('due date', data_obj.ds)
print('weight', data_obj.ws)
print('setup', data_obj.setup_time)
"""

optimizer.data_obj = data_obj 
solution = range(60)
# C = optimizer.evaluate(solution)
optimizer.fit(data_obj, iterations=50)
# print(C)