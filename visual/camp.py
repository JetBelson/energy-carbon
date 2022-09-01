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


if __name__ == "__main__":
    gt = load_data(file_name="../data/rf_data.csv")
    pd = load_data(file_name="../model/checkpoints/energy-carbon/9500_net_FFN.csv")
    # CO2/m2
    x = np.array([row[0] for row in gt])
    y = np.array([row[0] for row in pd])
    plt.scatter(x, y)
    plt.title("CO2/m2")
    plt.show()
    # eui
    x = np.array([row[1] for row in gt])
    y = np.array([row[1] for row in pd])
    plt.scatter(x, y)
    plt.title("eui")
    plt.show()