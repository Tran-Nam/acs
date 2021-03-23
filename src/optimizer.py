import numpy as np 
from time import time
from dataset import Dataset

class ACS:
    def __init__(self,
            ants=10,
            evaporation_rate=0.8,
            alpha=0.5,
            beta=0.5):
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha 
        self.beta = beta

        self.data_obj = None 
        self.set_of_available_nodes = None

        self.pheromone_matrix = None 
        self.probability_matrix = None
        self.heuristic_matrix = None

        self.visit_node = None

    def _reinstate_nodes(self):
        self.set_of_available_nodes = list(range(self.map.shape[0]))

    def _initialize(self):
        """
        initialize model parameter
            - pheromone
            - status of node
            - 
        """
        # assert self.map.shape[0] == self.map.shape[1], "Must equal num tasks"
        num_nodes = self.data_obj.setup_time.shape[0]
        self.pheromone_matrix = np.ones((num_nodes, num_nodes))
        self.pheromone_matrix[np.eye(num_nodes)==1] = 0
        self.set_of_available_nodes = list(range(num_nodes))

    def _update_probability(self):
        """
        update probability after evaporation and intensification
        """
        self.probability_matrix = (self.pheromone_matrix ** self.alpha) * \
            (self.heuristic_matrix ** self.beta)

    def _choose_next_node(self, from_node):
        """
        select next node base on probabillity and set of available nodes
        """        
        numerator = self.probability_matrix[from_node, self.set_of_available_nodes] # only use prob of remain nodes
        denominator = np.sum(numerator)
        probability = numerator / denominator
        next_node = np.random.choice(range(len(probability)), p=probability)
        return next_node


    def _update_pheromone(self):
        """
        evaporation
        """
        self.pheromone_matrix *= (1-self.evaporation_rate)
        
    def _remove_node(self, node):
        self.set_of_available_nodes.remove(node)

    def contruct_solution(self):
        pass

    def initialize_solution(self):
        pass

    def evaluate(self, solution):
        f_node = solution[0]
        t = self.data_obj.setup_time[f_node, f_node] + self.data_obj.ps[solution[f_node]]
        C = max(0, (t - self.data_obj.ds[f_node]))
        for i in range(1, len(solution)): # start from 1
            # print('iter', i)
            t += self.data_obj.setup_time[solution[i-1], solution[i]] # setup time 
            t += self.data_obj.ps[solution[i]]
            # print(t)
            C += max(0, (t - self.data_obj.ds[solution[i]]))
            # print(C)
        return C


    def fit(self, data_obj: Dataset, iterations=100):
        self.data_obj = data_obj
        s = time()
        for i in range(iterations):
            s_i = time()
            paths, path = [], []
            for ant in range(self.ants): # contruct solution for each ant
                self._initialize()
                current_node = np.random.choice(self.set_of_available_nodes) # choose random node to start
                start_node = current_node
                while True:
                    path.append(current_node)
                    self._remove_node(current_node)
                    if len(self.set_of_available_nodes) != 0:
                        current_node_index = self._choose_next_node(from_node=current_node)
                        current_node = self.set_of_available_nodes[current_node_index]
                    else:
                        break 
                self._reinstate_nodes()
                paths.append(path)
                path = []
            C = self.evaluate(solution=paths[0])
        print(C)

