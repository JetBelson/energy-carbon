import argparse
import os
from util import *
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--verbose', type=bool, default=True, help='log is verbal if true')
        self.parser.add_argument('--name', type=str, default='energy-carbon', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='FFN', help='model to use')
        self.parser.add_argument('--loss', type=str, default='MSE', help='loss to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')    
        self.parser.add_argument('--is_train', type=bool, default=False, help='train mode if true')
        self.parser.add_argument('--is_eval', type=bool, default=False, help='eval mode if true')
        self.parser.add_argument('--is_pred', type=bool, default=False, help='predict mode if true')
        
        # dataset specifics
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size for data loader')
        self.parser.add_argument('--dataroot', type=str, default='../data/data.csv', help='dataroot for train/eval/test')

        # optimizer specifics
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.02, help='initial learning rate for adam')
        self.parser.add_argument('--niter', type=int, default=100, help='update lr for each # iters')
        self.parser.add_argument('--niter_decay', type=float, default=100, help='# of iter to linearly decay learning rate to zero')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # set gpu
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt


if __name__ == "__main__":
    base_opt = BaseOptions()
    base_opt = base_opt.parse()
    print(base_opt.name)