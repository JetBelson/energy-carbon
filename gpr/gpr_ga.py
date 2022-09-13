import pandas as pd
import numpy as np
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, Exponentiation, Matern, RationalQuadratic
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
from sko.GA import GA, GA_TSP


wall = {1: 0.2, 2: 0.24, 3: 0.31, 4: 0.33}
roof = {1: 0.19, 2: 0.18, 3: 0.20}
glazing = {1: 1.65, 2: 2.16, 3: 1.52, 4: 1.16}


def load_data(file_name):
    """
    Load data from csv file
    @return df: pd dataframe.
    """
    ## Load init data
    df = pd.read_csv(
        file_name, index_col = 0, header = None,
        names = ['index', 'wall', 'roof', 'glazing', 'wwr', 'shading device', 'orientaition', 'energy cost', 'oclc', 'eec', 'tcpms']
    )
    ## Wash data
    df['orientaition'].replace('-', 0, inplace = True)
    df = df.astype(float)
    input012 = {'wall': wall, 'roof': roof, 'glazing': glazing}
    ## Replace type with number
    for key1 in input012:
        for key2 in input012[key1]:
            df[key1].replace(key2, input012[key1][key2], inplace = True)
    ## Calculate EUI
    df['eui'] = ((df['oclc']/(20*0.757)) / ((df['oclc'] + df['eec']) / df['tcpms']))
    return df


def train_eval_test(is_visual=False):
    """
    Train, eval, and test gpr model
    @param is_visual: plot test results if True 
    @return max_val: max value of indicators in train dataset
    @return gaussian_process: trained model
    """
    #### Load data from dataset
    ## Load train and eval data
    df = load_data(file_name="../data/train-eval.csv")
    df = df.astype(float)
    ## Load test data
    df2 = load_data(file_name="../data/test.csv")
    df2 = df2.astype(float)
    ## Normalization
    max_val = df.max()
    for key, value in zip(max_val.keys(), max_val):
        df[key] = df[key].divide(np.float32(value))
        df2[key] = df2[key].divide(np.float32(value))
    ## Divide data
    train_size = np.int32(0.8 * df.index.size)
    train_x = df.values[0:train_size,0:6]
    train_y = df.values[0:train_size,-2:]
    val_x = df.values[train_size:,0:6]
    val_y = df.values[train_size:,-2:]
    test_x = df2.values[:,0:6]
    test_y = df2.values[:,-2:]

    #### Train
    ## Build model
    kernel = (0.5 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(1e-3))
    gaussian_process = GaussianProcessRegressor(kernel = kernel, optimizer = 'fmin_l_bfgs_b', n_restarts_optimizer = 20)
    start_time = time.time()
    ## Train and eval
    gaussian_process.fit(train_x, train_y)
    print(f"Time for GaussianProcessRegressor fitting: {time.time() - start_time:.3f} seconds")
    pred_train_y = gaussian_process.predict(train_x)
    pred_val_y = gaussian_process.predict(val_x)
    ## Training and evaluating metric
    train_mse = mean_squared_error(train_y, pred_train_y )
    train_mae = mean_absolute_error(train_y, pred_train_y)
    val_mse = mean_squared_error(val_y, pred_val_y)
    val_mae = mean_absolute_error(val_y, pred_val_y)
    # Show
    print("train VS val (mean_squared_error): ", train_mse, "   ", val_mse)
    print("train VS val (mean_absolute_error): ", train_mae, "   ", val_mae)

    #### Test
    pred_test_y = gaussian_process.predict(test_x)
    ## Test metric
    scale = max_val[-2:].values.reshape(1,2).repeat(test_y.shape[0], axis = 0)
    test_mse = mean_squared_error(test_y * scale, pred_test_y * scale)
    test_mae = mean_absolute_error(test_y * scale, pred_test_y * scale)
    relative_error = (test_y - pred_test_y)/test_y
    test_max_relative_abs_error = np.max(np.abs(relative_error))
    test_mean_relative_abs_error = np.mean(np.abs(relative_error))
    ## Show
    print("test mean_squared_error: ", test_mse)
    print("test mean_absolute_error: ", test_mae)
    print("test max_relative_abs_error: ", test_max_relative_abs_error)
    print("test mean_relative_abs_error: ", test_mean_relative_abs_error)
    ## Visualization
    if is_visual:
        plt.plot(test_y[:,0] * max_val[-2], pred_test_y[:,0] * max_val[-2], 'k.')
        plt.plot(np.arange(1000,1600),np.arange(1000,1600), 'b-')
        plt.show()
        plt.plot(test_y[:,1] * max_val[-1], pred_test_y[:,1] * max_val[-1], 'k.')
        plt.plot(np.arange(60,100),np.arange(60,100), 'b-')
        plt.show()
    return gaussian_process, max_val


def gprFuncWrapper(model, max_val):
    """ Return opti function """
    #### FPS weights
    weights = [0, 1]

    def gprFunc(inputs: list):
        """
        Standard inputs: [{1,2,3,4}, {1,2,3}, {1,2,3,4}, [0, 30], {1,2}, [0, 360]]
        """
        assert inputs[0] in [1, 2, 3, 4]
        assert inputs[1] in [1, 2, 3]
        assert inputs[2] in [1, 2, 3, 4]
        assert 0<= inputs[3] <=30
        assert inputs[4] in [1, 2]
        assert 0<= inputs[5] <=360
        #### Normalization
        inputs = np.array([[
            wall[inputs[0]]/max_val[0],
            roof[inputs[1]]/max_val[1],
            glazing[inputs[2]]/max_val[2],
            inputs[3]/max_val[3],
            inputs[4]/max_val[4],
            inputs[5]/max_val[5]]])
        predicts = model.predict(inputs)
        predict = predicts[0, :]
        scale = max_val[-2:].values
        predict = predict*scale
        predict = weights[0]*predict[0] + weights[1]*predict[1]
        return predict

    return gprFunc


if __name__ == "__main__":
    gaussian_process, max_val = train_eval_test()
    func = gprFuncWrapper(model=gaussian_process, max_val=max_val)
    # print(func([1, 2, 4, 10, 2, 108]))
    ga = GA(func=func, n_dim=6, size_pop=100, max_iter=500, prob_mut=0.001, 
            lb=[1, 1, 1, 10, 1, 0], 
            ub=[4, 3, 4, 30, 2, 359], 
            precision=[1, 1, 1, 1e-3, 1, 1])
    best_x, best_y = ga.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()
