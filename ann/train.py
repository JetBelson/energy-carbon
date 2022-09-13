import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict

from opt import BaseOptions
from ffn_model import CreateModel
from ec_data_loader import CreateDataLoader
from util import log


if __name__ == "__main__":
    opt = BaseOptions().parse()
    start_epoch, epoch_iter = 1, 0
    #### create train and eval dataloader
    data_loader = CreateDataLoader(opt=opt, data_type="train")
    train_loader = data_loader.load_data()
    data_loader = CreateDataLoader(opt=opt, data_type="eval")
    eval_loader = data_loader.load_data()

    #### TODO add continue train option

    #### create model
    model = CreateModel(opt=opt)

    #### train and eval model
    # TODO add some parameters to options
    fo = open("{}/{}/loss.txt".format(opt.checkpoints_dir, opt.name), "w")
    fo.write("Epoch, Loss\n")
    total_steps = 0
    local_loss = []
    for epoch in range(10000):
        epoch_loss = []
        for i, train_data in enumerate(train_loader):
            total_steps += 1
            ######### Forward Pass ##########
            loss = model(train_data)
            local_loss.append(loss.item())
            epoch_loss.append(loss.item()) 
            ######### Backward Pass ##########
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            ######### Display results ##########
            ### print out errors every 50 steps
            if (total_steps+1) % 50 == 0:
                print("epoch iter: {},  current step: {}, average loss: {}".format(epoch, i, sum(local_loss)/len(local_loss)))
                local_loss = []
        
        ######### Log errors every epoch ##########
        log(file=fo, msg="{},{}".format(epoch, sum(epoch_loss)/len(epoch_loss)), is_print=False)

        ####### Evaluate model every 100 epochs #######
        if epoch % 100 == 0:
            print("******************************************************")
            print("evaluating model at the end of epoch {}, iters {}".format(epoch, total_steps))
            model.eval()
            eval_loss, batches = 0, 0
            for i, eval_data in enumerate(eval_loader):
                ######### Forward Pass ##########
                loss = model(train_data)
                eval_loss += loss.item()
                batches += 1
            model.train()
            print("Eval loss: {}".format(eval_loss/batches))
            print("******************************************************")

        ####### Save model every 500 epochs #######
        if (epoch+1) % 500 == 0:
            print("saving model at the end of epoch {}, iters {}".format(epoch, total_steps))
            model.save(epoch)
        if epoch % opt.niter == 1:
            model.update_learning_rate()
