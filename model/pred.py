import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from tqdm import tqdm
from util import log
from opt import BaseOptions
from ffn_model import CreateModel
from ec_data_loader import CreateDataLoader


if __name__ == "__main__":
    opt = BaseOptions().parse()
    #### create train and eval dataloader
    data_loader = CreateDataLoader(opt=opt, data_type="predict")
    pred_loader = data_loader.load_data()

    #### create model 
    model = CreateModel(opt=opt)
    model.load(which_epoch=9500)
    
    #### log file
    save_filename = '%s_net_%s.csv' % (9500, "FFN")
    pred_file = open(os.path.join(model.save_dir, save_filename), "w")
    log(file=pred_file, msg="pred-AvgCO2,pred-EUI")
    start_time = time.time()
    for i, pred_data in tqdm(enumerate(pred_loader)):
        ######### Forward Pass ##########
        predicts = model.inference(pred_data)
        predicts.tolist()
        for row in predicts:
            log(file=pred_file, msg="{},{}".format(int(row[0]), int(row[1])))
    ######### Display results ##########
    end_time = time.time()
    print("Total time consumed: {}".format(end_time-start_time))
