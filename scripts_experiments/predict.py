import os
from options.base_options import BaseOptions
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from data.rhcaa_predict import rhcaa_diene

from icecream import ic

def predict() -> None:

    opt = BaseOptions().parse()

    # Get the current working directory
    current_dir = os.getcwd()
    
    # Load the final test set
    dataset =rhcaa_diene(opt, opt.filename_predict, opt.mol_cols, opt.root_predict)
    loader = DataLoader(dataset, shuffle=False)

    experiments_gnn = os.path.join(current_dir, opt.log_dir_results, 'final_test', 'results_GNN')

    predictions_all = pd.DataFrame()

    for outer in range(1, opt.folds+1):
        print('Analysing models trained using as test set {}'.format(outer))
        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            print('Analysing models trained using as validation set {}'.format(real_inner))

            model_dir = os.path.join(current_dir, opt.log_dir_results, opt.predict_model, 'results_GNN', f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            model = torch.load(model_dir+'/model.pth')
            model_params = torch.load(model_dir+'/model_params.pth')
            model.load_state_dict(model_params)

            for batch in loader:

                batch = batch.to('cpu')
                out, emb = model(batch, True)

                y_pred.append(out.cpu().detach().numpy())
                idx.append(batch.idx.cpu().detach().numpy())
                embeddings.append(emb.detach().numpy())

            y_pred = np.concatenate(y_pred, axis=0).ravel()
            y_true = np.concatenate(y_true, axis=0).ravel()
            idx = np.concatenate(idx, axis=0).ravel()
            embeddings = np.concatenate(embeddings, axis=0)

            embeddings = pd.DataFrame(embeddings)

            embeddings['ddG_exp'] = y_true
            embeddings['ddG_pred'] = y_pred
            embeddings['index'] = idx

            predictions_all = pd.concat([predictions_all, embeddings], axis=0)
            

        
