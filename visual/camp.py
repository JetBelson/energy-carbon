"""
Visualize prediction and ground truth
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_name):
    content = []
    with open(file_name, "r") as rf:
        lines = rf.readlines()
        lines = [line.strip().split(",") for line in lines]
    for line in lines[1:]:
        content.append([int(line[0]), int(line[1])])
    return content


def model_pf(gt_file="../data/rf_test.csv", pd_file="../model/results/energy-carbon/9999_net_FFN.csv", is_show=True):
    gt = load_data(gt_file)
    pd = load_data(pd_file)
    #### Load data
    # CO2/m2
    x_co2 = np.array([row[0] for row in gt])
    y_co2 = np.array([row[0] for row in pd])
    # eui
    x_eui = np.array([row[1] for row in gt])
    y_eui = np.array([row[1] for row in pd])

    #### Visualize results
    if is_show:
        plt.scatter(x_co2, y_co2)
        plt.title("CO2/m2")
        plt.show()
        plt.scatter(x_eui, y_eui)
        plt.title("eui")
        plt.show()
    #### log book
    log_name = pd_file.split("/")[-1].split(".")[0]
    fo = open("./error/"+log_name+".txt", "w")
    fo.write("Indicator, Avg absolute error, Avg related error, Max absolute error, Max related error\n")
    #### calculate results
    ab_error_co2 = np.abs(x_co2-y_co2)
    avg_ab_error_co2 = np.average(ab_error_co2)
    rl_error_co2 = ab_error_co2 / x_co2
    avg_rl_error_co2 = np.average(rl_error_co2)
    max_ab_error_co2 = np.max(ab_error_co2)
    max_rl_error_co2 = np.max(rl_error_co2)
    fo.write("C02,{:.2},{:.2},{:.2},{:.2}\n".format(
        avg_ab_error_co2.astype(float),
        avg_rl_error_co2.astype(float),
        max_ab_error_co2.astype(float),
        max_rl_error_co2.astype(float)
    ))
    ab_error_eui = np.abs(x_eui-y_eui)
    avg_ab_error_eui = np.average(ab_error_eui)
    rl_error_eui = ab_error_eui / x_eui
    avg_rl_error_eui = np.average(rl_error_eui)
    max_ab_error_eui = np.max(ab_error_eui)
    max_rl_error_eui = np.max(rl_error_eui)
    fo.write("EUI,{:.2},{:.2},{:.2},{:.2}\n".format(
        avg_ab_error_eui.astype(float),
        avg_rl_error_eui.astype(float),
        max_ab_error_eui.astype(float),
        max_rl_error_eui.astype(float)
    ))


if __name__ == "__main__":
    model_pf(is_show=False)