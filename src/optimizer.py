import numpy as np 
from time import time
import matplotlib.pyplot as plt
from dataset import Dataset
np.random.seed(20202020)

class ACS:
    def __init__(self,
            ants=10,
            evaporation_rate=0.2,
            intensification=1,
            alpha=1,
            beta=0):
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.intensification = intensification
        self.alpha = alpha 
        self.beta = beta

        self.data_obj = None 
        self.set_of_available_nodes = None

        self.pheromone_matrix = None 
        self.probability_matrix = None
        self.heuristic_matrix = None

        self.best_loss = []

    def _reinstate_nodes(self):
        self.set_of_available_nodes = list(range(self.data_obj.setup_time.shape[0]))

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

        # self.heuristic_matrix = 1 / (self.data_obj.setup_time + 1e-6)
        # self.heuristic_matrix = np.ones((num_nodes, num_nodes))
        # self.heuristic_matrix[np.eye(num_nodes)==1] = 1 / self.data_obj.ds
        self.heuristic_matrix = 1 / self.data_obj.ds
        print('HEURISTIC MATRIX')
        print(self.heuristic_matrix.round(3))
        self.probability_matrix = (self.pheromone_matrix ** self.alpha) * \
            (self.heuristic_matrix ** self.beta)
        print('PROBABILITY MATRIX')
        print(self.probability_matrix.round(3))
        print('='*10)
        # input()

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
        # print('choose next node', from_node, numerator.round(3))
        denominator = np.sum(numerator)
        probability = numerator / denominator
        next_node = np.random.choice(range(len(probability)), p=probability)
        print('from node', from_node, '- next node', next_node)
        return next_node


    def _update_pheromone(self, solutions, scores, current_best_loss):
        """
        evaporation
        """
        # print('update pheromone', self.pheromone_matrix)
        self.pheromone_matrix = self.pheromone_matrix * (1-self.evaporation_rate) + self.evaporation_rate / current_best_loss
        for solution, score in zip(solutions, scores):
            coord_i, coord_j = [], []
            for i in range(len(solution) - 1):
                coord_i.append(solution[i])
                coord_j.append(solution[i+1])
            self.pheromone_matrix[coord_i, coord_j] += 1 / score
        # print('after pheromone', self.pheromone_matrix)
        
    def _intensify(self, coord):
        i, j = coord 
        self.pheromone_matrix[i, j] += self.intensification
        
    def _remove_node(self, node):
        self.set_of_available_nodes.remove(node)

    # def evaluate(self, solution):
    #     f_node = solution[0]
    #     t = self.data_obj.setup_time[f_node, f_node] + self.data_obj.ps[solution[f_node]]
    #     C = self.data_obj.ws[f_node] * max(0, (t - self.data_obj.ds[f_node]))
    #     for i in range(1, len(solution)): # start from 1
    #         # print('iter', i)
    #         t += self.data_obj.setup_time[solution[i-1], solution[i]] # setup time 
    #         t += self.data_obj.ps[solution[i]]
    #         # print(t)
    #         C += self.data_obj.ws[solution[i]] * max(0, (t - self.data_obj.ds[solution[i]]))
    #         # print(C)
    #     return C

    def _evaluate(self, solutions):
        scores, coord_is, coord_js = [], [], []
        for i, solution in enumerate(solutions):
            coord_i, coord_j = [], []
            score = 0
            for i in range(len(solution)):
                # print(solution[i])
                if i < len(solution) - 1:
                    coord_i.append(solution[i])
                    coord_j.append(solution[i+1])
                if i==0: #start node
                    t = self.data_obj.setup_time[solution[i], solution[i]] + self.data_obj.ps[solution[i]]
                else:
                    t += self.data_obj.setup_time[solution[i-1], solution[i]] + self.data_obj.ps[solution[i]]
                # print('t =', t)
                # print(self.data_obj.ds[solution[i]])
                score += self.data_obj.ws[solution[i]] * max(0, (t - self.data_obj.ds[solution[i]]))
                # print(score)
            scores.append(score)
            coord_is.append(coord_i)
            coord_js.append(coord_j)
        idx = np.argmin(scores)
        print('EVALUATE!')
        for solution, score in zip(solutions, scores):
            print(solution, ' - ', score)
        return (coord_is[idx], coord_js[idx]), solutions[idx], scores[idx], scores

    def fit(self, data_obj: Dataset, iterations=100):
        self.data_obj = data_obj
        self._initialize()
        s = time()
        best_path_list = []
        for i in range(iterations):
            print('='*20)
            print('iter:', i)
            # print('PHEROMONE MATRIX')
            # print(self.pheromone_matrix.round(2))
            print('PROBABILITY MATRIX')
            print(self.probability_matrix.round(3))
            # print()
            s_i = time()
            paths, path = [], []
            for ant in range(self.ants): # contruct solution for each ant
                print('-'*5)
                print('ant:', ant)
                # self._update_probability()
                current_node = np.random.choice(self.set_of_available_nodes) # choose random node to start
                start_node = current_node
                # print('start node', start_node)
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
            coord, solution, C, scores = self._evaluate(paths)
            best_path_list.append(solution)
            print('ALL SOLUTION:', paths)
            print('BEST SOLUTION:', solution)
            self.best_loss.append(C)
            # print('iter:', i, 'loss:', C)
            print('BEFORE UPDATE PHEROMONE')
            print(self.pheromone_matrix.round(3))
            self._update_pheromone(paths, scores, C)
            print('BEFORE INTENSIFY')
            print(self.pheromone_matrix.round(3))
            self._intensify(coord)
            print('AFTER INTENSIFY')
            print(self.pheromone_matrix.round(3))
            self._update_probability()
            # input()
        print('*'*20)
        for path, loss in zip(best_path_list, self.best_loss):
            print(path, ':', loss)
        # self.plot()
    def plot(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(self.best_loss, label='Best run')
        plt.show()
                

