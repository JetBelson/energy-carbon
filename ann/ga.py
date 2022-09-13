import numpy as np
from sko.GA import GA, GA_TSP
import pandas as pd
import matplotlib.pyplot as plt
import torch

from opt import BaseOptions
from ffn_model import CreateModel
from ec_data_loader import CreateDataLoader


def ecANNWrapper(model):
    #### TFS weights
    weights = [0, 1]
    
    def ecANN(inputs):
        """
        Standard inputs: [{1,2,3,4}, {1,2,3}, {1,2,3,4}, [0, 30], {1,2}, [0, 360]]
        """
        assert inputs[0] in [1, 2, 3, 4]
        assert inputs[1] in [1, 2, 3]
        assert inputs[2] in [1, 2, 3, 4]
        assert 0<= inputs[3] <=30
        assert inputs[4] in [1, 2]
        assert 0<= inputs[5] <=360

        wall = {1: 0.2, 2: 0.24, 3: 0.31, 4: 0.33}
        roof = {1: 0.19, 2: 0.18, 3: 0.20}
        glazing = {1: 1.65, 2: 2.16, 3: 1.52, 4: 1.16}
        inputs = [wall[inputs[0]], roof[inputs[1]], glazing[inputs[2]], inputs[3], inputs[4], inputs[5]]
        inputs = torch.Tensor([inputs])
        predicts = model.predict(inputs)
        predicts.tolist()
        predict = predicts[0]
        predict = weights[0]*predict[0] + weights[1]*predict[1]
        return predict.detach().numpy()
    
    return ecANN

if __name__ == "__main__":
    opt = BaseOptions().parse()
    model = CreateModel(opt=opt)
    model.load(which_epoch=9999)
    func = ecANNWrapper(model=model)
    # print(func([2, 2, 4, 10, 2, 108]))
    ga = GA(func=func, n_dim=6, size_pop=100, max_iter=500, prob_mut=0.001, 
            lb=[1, 1, 1, 10, 1, 0], 
            ub=[4, 3, 4, 30, 2, 359], 
            precision=[1, 1, 1, 1e-1, 1, 1])
    best_x, best_y = ga.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()
