import os
import sys
import joblib
import torch
from torch_geometric.loader import DataLoader
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.utils_model import tml_report, network_outer_report, network_report, calculate_morgan_fingerprints
from options.base_options import BaseOptions
from data.rhcaa import rhcaa_diene
from data.biaryl import rhcaa_biaryl
from data.general_reaction import reaction_representation
from data.hypervalent_iodine import hypervalent_graph


from icecream import ic

def predict_final_test(opt) -> None:


    # Get the current working directory
    current_dir = os.getcwd()
    
    # Load the final test set

    # Create the dataset
    if opt.filename_final_test =='biaryl.csv':
        data = rhcaa_biaryl(opt, opt.filename_final_test, opt.mol_cols, opt.root_final_test, include_fold=False)
    elif opt.filename_final_test == 'N_S_acetal.csv' or opt.filename == 'asym_hydrogenation.csv':
        data = reaction_representation(opt, opt.filename_final_test, opt.mol_cols, opt.root_final_test, include_fold=False)
    elif opt.filename_final_test == 'hypervalent_iodine.csv':
        data = hypervalent_graph(opt, opt.filename_final_test, opt.mol_cols, opt.root_final_test, include_fold=False)
    else:
        data = rhcaa_diene(opt, opt.filename_final_test, opt.mol_cols, opt.root_final_test, include_fold=False)


    test_loader = DataLoader(data, shuffle=False)



    experiments_gnn = os.path.join(current_dir, opt.log_dir_results, opt.filename_final_test[:-4], 'test','results_GNN')

    for outer in range(1, opt.folds+1):
        print('Analysing models trained using as test set {}'.format(outer))
        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            print('Analysing models trained using as validation set {}'.format(real_inner))

            model_dir = os.path.join(current_dir, opt.log_dir_results, opt.filename[:-4],'learning', 'results_GNN', f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            model = torch.load(model_dir+'/model.pth', weights_only=False)
            model_params = torch.load(model_dir+'/model_params.pth', weights_only=True)
            train_loader = torch.load(model_dir+'/train_loader.pth', weights_only=False)
            val_loader = torch.load(model_dir+'/val_loader.pth', weights_only=False)

            network_report(log_dir=experiments_gnn,
                           loaders=(train_loader, val_loader, test_loader),
                           outer=outer,
                           inner=real_inner,
                           loss_lists=[None, None, None],
                           model=model,
                           model_params=model_params,
                           best_epoch=None,
                           save_all=False)
                                    
        network_outer_report(log_dir=f"{experiments_gnn}/Fold_{outer}_test_set/", 
                             outer=outer,
                             folds=opt.folds,)
        

        
if __name__ == '__main__':
    opt = BaseOptions().parse()
    predict_final_test(opt)