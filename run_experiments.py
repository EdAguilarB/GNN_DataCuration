import os
from scripts_experiments.train_GNN import train_network_nested_cv
from options.base_options import BaseOptions
import os

def run_all_exp():

    opt = BaseOptions().parse()

    if opt.train_GNN:
        if not os.path.exists(os.path.join(opt.log_dir_results, opt.filename[:-4], 'results_GNN')):
            train_network_nested_cv(opt)
        else:
            print('GNN model has already been trained')
    

if __name__ == '__main__':
    run_all_exp()

