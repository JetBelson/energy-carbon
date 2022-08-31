import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torch


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return "BaseDataset"

    def initialize(self, opt):
        pass


class ECDataset(BaseDataset):
    def initialize(self, opt, data_type="train"):
        """
        initialize dataset according to options
        @param: opt, class BaseOptions 
        @param: data_type, train, eval, predict
        """
        self.opt = opt
        self.root = opt.dataroot
        self.data_type = data_type

        ### load data from root
        with open(self.root, "r") as rf:
            lines = rf.readlines()
        lines = [line.strip() for line in lines]
        data_size = len(lines)

        if data_type == "train":
            train_data_size = int(data_size*0.8)
            self.data = lines[:train_data_size]

        if data_type == "eval":
            eval_data_size = int(data_size*0.2)
            self.data = lines[-eval_data_size:]

        if data_type == "predict":
            self.data = lines

        ### U-values of different type
        self.wall = {1: 0.2, 2: 0.24, 3: 0.31, 4: 0.33}
        self.roof = {1: 0.19, 2: 0.18, 3: 0.20}
        self.glazing = {1: 1.65, 2: 2.16, 3: 1.52, 4: 1.16}

    def name(self):
        return "ECDataset"
    
    def __getitem__(self, index):
        """
        @return: data_dict, {"x": [], "y": []}
        """
        item = self.data[index]
        item = item.split(",")
        item = self.num(item=item)
        data_dict = {}
        if self.data_type=="train" or self.data_type=="eval":
            data_dict["x"] = torch.Tensor(item[1:7]).unsqueeze(-1)
            data_dict["y"] = item[-1]
        elif self.data_type == "predict":
            data_dict["x"] = torch.Tensor(item[1:7]).unsqueeze(-1)
            data_dict["y"] = None
        return data_dict

    def __len__(self):
        return len(self.data)

    def num(self, item):
        """
        Transform item into number
        @param item, list of string
        @return num_item, string to number, "-" --> 0 
        """
        num_item = []
        for factor in item:
            if factor == "-":
                num_item.append(0)
            else:
                num_item.append(int(factor))
        try:
            num_item[1] = self.wall[num_item[1]]
            num_item[2] = self.roof[num_item[2]]
            num_item[3] = self.glazing[num_item[3]]
        except:
            print(num_item[1])
            print(num_item[2])
            print(num_item[3])
            raise KeyError
        return num_item


    