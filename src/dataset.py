import os 
import numpy as np

class Dataset():
    def __init__(self, path):
        self.extract(path)
        pass 
    def extract(self, path):
        with open(path, 'r') as f:
            data = f.read().splitlines()
        p_id = data.index('Process Times:')
        w_id = data.index('Weights:')
        d_id = data.index('Duedates:')
        s_id = data.index('Setup Times:')
        e_id = data.index('End Problem Specification')

        ps = data[p_id+1: w_id]
        ws = data[w_id+1: d_id]
        ds = data[d_id+1: s_id]
        ss = data[s_id+1: e_id]
        
        self.ps = np.array([int(p) for p in ps])
        self.ws = np.array([int(w) for w in ws])
        self.ds = np.array([int(d) for d in ds])

        self.n_tasks = len(ps)
        # self.begin_time = np.zeros((self.n_tasks, 1))
        self.setup_time = np.zeros((self.n_tasks, self.n_tasks))
        for line in ss:
            s_task, e_task, time = [int(l) for l in line.split('\t')]
            if s_task==-1:
                # self.begin_time[e_task] = time 
                self.setup_time[e_task, e_task] = time
            else:
                self.setup_time[s_task, e_task] = time 
        # print(self.begin_time)
        # print(self.setup_time)

# path = '../data/wtsds-instances/wt_sds_1.instance'
# D = Dataset(path)
