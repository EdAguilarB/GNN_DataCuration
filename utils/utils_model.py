import torch
import os
import csv
import re
import numpy as np
from datetime import date, datetime
from copy import copy, deepcopy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    accuracy_score, precision_score, recall_score, f1_score
from math import sqrt
from utils.plot_utils import *
from collections import defaultdict
from icecream import ic


def calculate_metrics(y_true:list, y_predicted: list,  task = 'r'):

    if task == 'r':
        r2 = r2_score(y_true=y_true, y_pred=y_predicted)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
        rmse = sqrt(mean_squared_error(y_true=y_true, y_pred=y_predicted))  
        error = [(y_predicted[i]-y_true[i]) for i in range(len(y_true))]
        prctg_error = [ abs(error[i] / y_true[i]) for i in range(len(error)) if y_true[i] != 0]
        mbe = np.mean(error)
        mape = np.mean(prctg_error)
        error_std = np.std(error)
        metrics = [r2, mae, rmse, mbe, mape, error_std]
        metrics_names = ['R2', 'MAE', 'RMSE', 'Mean Bias Error', 'Mean Absolute Percentage Error', 'Error Standard Deviation']

    elif task == 'c':
        accuracy = accuracy_score(y_true=y_true, y_pred=y_predicted)
        precision = precision_score(y_true=y_true, y_pred=y_predicted)
        recall = recall_score(y_true=y_true, y_pred=y_predicted)
        f1 = f1_score(y_true=y_true, y_pred=y_predicted)
        metrics = [accuracy, precision, recall, f1]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    return np.array(metrics), metrics_names

######################################
######################################
######################################
###########  GNN functions ###########
######################################
######################################
######################################

def train_network(model, train_loader, device):

    train_loss = 0
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        model.optimizer.zero_grad()
        out = model(batch)
        loss = torch.sqrt(model.loss(out, torch.unsqueeze(batch.y, dim=1)))
        loss.backward()
        model.optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    return train_loss / len(train_loader.dataset)

def eval_network(model, loader, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss += torch.sqrt(model.loss(out, torch.unsqueeze(batch.y, dim = 1) )).item() * batch.num_graphs
    return loss / len(loader.dataset)


def predict_network(model, loader, return_emb = False, target_var=None):
    model.to('cpu')
    model.eval()

    y_pred, y_true, idx, embeddings = [], [], [], []

    for batch in loader:
        batch = batch.to('cpu')
        out, emb = model(batch, True)

        y_pred.append(out.cpu().detach().numpy())
        y_true.append(batch.y.cpu().detach().numpy())
        idx.append(batch.idx.cpu().detach().numpy())
        embeddings.append(emb.detach().numpy())

    y_pred = np.concatenate(y_pred, axis=0).ravel()
    y_true = np.concatenate(y_true, axis=0).ravel()
    idx = np.concatenate(idx, axis=0).ravel()
    embeddings = np.concatenate(embeddings, axis=0)

    embeddings = pd.DataFrame(embeddings)

    embeddings[f'{target_var}_exp'] = y_true
    embeddings[f'{target_var}_pred'] = y_pred
    embeddings['index'] = idx

    if return_emb == True:
        return y_pred, y_true, idx, embeddings
    else:
        return y_pred, y_true, idx


def network_report(log_dir,
                   loaders,
                   outer,
                   inner, 
                   loss_lists,
                   save_all,
                   model, 
                   model_params,
                   best_epoch, 
                   opt):


    #1) Create a directory to store the results
    log_dir = "{}/Fold_{}_test_set/Fold_{}_val_set".format(log_dir, outer, inner)
    os.makedirs(log_dir, exist_ok=True)

    #2) Time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    #3) Unfold loaders and save loaders and model
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    N_train, N_val, N_test = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    N_tot = N_train + N_val + N_test     
    if save_all == True:
        torch.save(train_loader, "{}/train_loader.pth".format(log_dir))
        torch.save(val_loader, "{}/val_loader.pth".format(log_dir))
        torch.save(model, "{}/model.pth".format(log_dir))
        torch.save(model_params, "{}/model_params.pth".format(log_dir))
    torch.save(test_loader, "{}/test_loader.pth".format(log_dir)) 
    loss_function = 'RMSE_%'

    #4) loss trend during training
    train_list = loss_lists[0]
    val_list = loss_lists[1] 
    test_list = loss_lists[2]
    if train_list is not None and val_list is not None and test_list is not None:
        with open('{}/{}.csv'.format(log_dir, 'learning_process'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Train_{}".format(loss_function), "Val_{}".format(loss_function), "Test_{}".format(loss_function)])
            for i in range(len(train_list)):
                writer.writerow([(i+1)*5, train_list[i], val_list[i], test_list[i]])
        create_training_plot(df='{}/{}.csv'.format(log_dir, 'learning_process'), save_path='{}'.format(log_dir))


    #5) Start writting report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN TRAINING AND PERFORMANCE\n")
    file1.write("Best epoch: {}\n".format(best_epoch))
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")

    model.load_state_dict(model_params)

    y_pred, y_true, idx, emb_train = predict_network(model, train_loader, True, opt.target_var)
    emb_train['set'] = 'training'
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    
    file1.write("***************\n")
    y_pred, y_true, idx, emb_val = predict_network(model, val_loader, True, opt.target_var)
    emb_val['set'] = 'val'
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Validation set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("***************\n")

    y_pred, y_true, idx, emb_test = predict_network(model, test_loader, True, opt.target_var)
    emb_test['set'] = 'test'

    emb_all = pd.concat([emb_train, emb_val, emb_test], axis=0)

    plot_tsne_with_subsets(data_df=emb_all, feature_columns=[i for i in range(128)], color_column=f'{opt.target_var}_exp', set_column='set', fig_name='tsne_emb_exp', save_path=log_dir)
    #plot_tsne_with_subsets(data_df=emb_all, feature_columns=[i for i in range(128)], color_column='ddG_pred', set_column='set', fig_name='tsne_emb_pred', save_path=log_dir)
    emb_all.to_csv("{}/embeddings.csv".format(log_dir))

    pd.DataFrame({f'real_{opt.target_var}': y_true, f'predicted_{opt.target_var}': y_pred, 'index': idx}).to_csv("{}/predictions_test_set.csv".format(log_dir))

    face_pred = np.where(y_pred > 50, 1, 0)
    face_true = np.where(y_true > 50, 1, 0)
    metrics, metrics_names = calculate_metrics(face_true, face_pred, task = 'c')

    correct_side_add = face_pred == face_true

    file1.write("Test set\n")
    file1.write("Set size = {}\n".format(N_test))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))
    
    #error = abs(y_pred-y_true)
    #y_true = y_true[correct_side_add]
    #y_pred = y_pred[correct_side_add]
    #idx = idx[correct_side_add]


    file1.write("Test Set Total Correct Face of Addition Predictions = {}\n".format(np.sum(correct_side_add)))

    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("---------------------------------------------------------\n")

    create_st_parity_plot(real = y_true, predicted = y_pred, figure_name = 'outer_{}_inner_{}'.format(outer, inner), save_path = "{}".format(log_dir))
    #create_it_parity_plot(real = y_true, predicted = y_pred, index = idx, figure_name='outer_{}_inner_{}.html'.format(outer, inner), save_path="{}".format(log_dir))

    file1.write("OUTLIERS (TEST SET)\n")
    error_test = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))]
    abs_error_test = [abs(error_test[i]) for i in range(len(y_pred))]
    std_error_test = np.std(error_test)

    outliers_list, outliers_error_list, index_list = [], [], []

    counter = 0

    for sample in range(len(y_pred)):
        if abs_error_test[sample] >= 3 * std_error_test:  
            counter += 1
            outliers_list.append(idx[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write("0{}) {}    Error: {:.2f} {}    (index={})\n".format(counter, idx[sample], error_test[sample], opt.target_var_units, idx[sample]))
            else:
                file1.write("{}) {}    Error: {:.2f} {}    (index={})\n".format(counter, idx[sample], error_test[sample], opt.target_var_units, idx[sample]))

    file1.close()

    return 'Report saved in {}'.format(log_dir)


import os
import re
import numpy as np
from collections import defaultdict
from icecream import ic

def network_outer_report(log_dir: str, outer: int, folds: int, threshold: float = 30.0):
    accuracy, precision, recall, r2, mae, rmse = [], [], [], [], [], []
    outlier_counts = defaultdict(int)  # Moved here to accumulate across all folds

    # List of performance files for each fold except the current outer fold
    files = [log_dir + f'Fold_{i}_val_set/performance.txt' for i in range(1, folds + 1) if i != outer]

    # Define regular expressions to match metric and outlier lines
    accuracy_pattern = re.compile(r"Accuracy = (\d+\.\d+)")
    precision_pattern = re.compile(r"Precision = (\d+\.\d+)")
    recall_pattern = re.compile(r"Recall = (\d+\.\d+)")
    r2_pattern = re.compile(r"R2 = (\d+\.\d+)")
    mae_pattern = re.compile(r"MAE = (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE = (\d+\.\d+)")
    outlier_pattern = re.compile(r"\d+\)\s+(\d+)\s+Error:\s+(-?\d+\.\d+)\s+\S+\s+\(index=(\d+)\)")

    for file in files:
        with open(file, 'r') as f:
            content = f.read()

        # Split content by '*' to separate training, validation, and test sets
        sets = content.split('*')

        for set_content in sets:
            # Check if "Test set" is in the set content
            if "Test set" in set_content:
                # Extract metric values using regular expressions
                if (accuracy_match := accuracy_pattern.search(set_content)):
                    accuracy.append(float(accuracy_match.group(1)))
                if (precision_match := precision_pattern.search(set_content)):
                    precision.append(float(precision_match.group(1)))
                if (recall_match := recall_pattern.search(set_content)):
                    recall.append(float(recall_match.group(1)))
                if (r2_match := r2_pattern.search(set_content)):
                    r2.append(float(r2_match.group(1)) if r2_match else 0)
                if (mae_match := mae_pattern.search(set_content)):
                    mae.append(float(mae_match.group(1)))
                if (rmse_match := rmse_pattern.search(set_content)):
                    rmse.append(float(rmse_match.group(1)))

                # Search for outliers in the "OUTLIERS (TEST SET)" section
                outliers_section = set_content.split("OUTLIERS (TEST SET)", 1)[-1]
                for outlier_match in outlier_pattern.finditer(outliers_section):
                    ic(outlier_match)
                    error = abs(float(outlier_match.group(2)))  # Get absolute error percentage
                    ic(error)
                    index = int(outlier_match.group(3))
                    ic(index)
                    if error > threshold:  # Use threshold parameter to filter errors
                        outlier_counts[index] += 1
                        ic(outlier_counts[index])

    ic(outlier_counts)  # To verify final counts across folds

    # Calculate mean and standard deviation for each metric
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    precision_mean = np.mean(precision)
    precision_std = np.std(precision)
    recall_mean = np.mean(recall)
    recall_std = np.std(recall)
    r2_mean = np.mean(r2)
    r2_std = np.std(r2)
    mae_mean = np.mean(mae)
    mae_std = np.std(mae)
    rmse_mean = np.mean(rmse)
    rmse_std = np.std(rmse)

    # Write the results to the file
    report_path = f"{log_dir}/performance_outer_test_fold{outer}.txt"
    with open(report_path, "w") as file1:
        file1.write("---------------------------------------------------------\n")
        file1.write("Test Set Metrics (mean ± std)\n")
        file1.write(f"Accuracy: {accuracy_mean:.3f} ± {accuracy_std:.3f}\n")
        file1.write(f"Precision: {precision_mean:.3f} ± {precision_std:.3f}\n")
        file1.write(f"Recall: {recall_mean:.3f} ± {recall_std:.3f}\n")
        file1.write(f"R2: {r2_mean:.3f} ± {r2_std:.3f}\n")
        file1.write(f"MAE: {mae_mean:.3f} ± {mae_std:.3f}\n")
        file1.write(f"RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}\n")
        file1.write("---------------------------------------------------------\n")

        # Write outlier summary
        file1.write("Outliers:\n")
        for index, count in sorted(outlier_counts.items()):
            file1.write(f"Index {index}: {count} times\n")

    return f'Report saved in {report_path}'

def extract_metrics(file):

    metrics = {'Accuracy': None, 'Precision': None, 'Recall': None, 'R2': None, 'MAE': None, 'RMSE': None}

    with open(file, 'r') as file:
            content = file.read()

    # Define regular expressions to match metric lines
    accuracy_pattern = re.compile(r"Accuracy: (\d+\.\d+) ± (\d+\.\d+)")
    precision_pattern = re.compile(r"Precision: (\d+\.\d+) ± (\d+\.\d+)")
    recall_pattern = re.compile(r"Recall: (\d+\.\d+) ± (\d+\.\d+)")
    r2_pattern = re.compile(r"R2: (\d+\.\d+) ± (\d+\.\d+)")
    mae_pattern = re.compile(r"MAE: (\d+\.\d+) ± (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE: (\d+\.\d+) ± (\d+\.\d+)")

    accuracy_match = accuracy_pattern.search(content)
    precision_match = precision_pattern.search(content)
    recall_match = recall_pattern.search(content)
    r2_match = r2_pattern.search(content)
    mae_match = mae_pattern.search(content)
    rmse_match = rmse_pattern.search(content)

    # Update the metrics dictionary with extracted values
    if accuracy_match:
        metrics['Accuracy'] = {'mean': float(accuracy_match.group(1)), 'std': float(accuracy_match.group(2))}
    if precision_match:
        metrics['Precision'] = {'mean': float(precision_match.group(1)), 'std': float(precision_match.group(2))}
    if recall_match:
        metrics['Recall'] = {'mean': float(recall_match.group(1)), 'std': float(recall_match.group(2))}
    if r2_match:
        metrics['R2'] = {'mean': float(r2_match.group(1)), 'std': float(r2_match.group(2))}
    if mae_match:
        metrics['MAE'] = {'mean': float(mae_match.group(1)), 'std': float(mae_match.group(2))}
    if rmse_match:
        metrics['RMSE'] = {'mean': float(rmse_match.group(1)), 'std': float(rmse_match.group(2))}

    return metrics

    

def split_data(df:pd.DataFrame):

    '''
    splits a dataset in a given quantity of folds
    '''
        
    for outer in np.unique(df['fold']):
        proxy = copy(df)
        test = proxy[proxy['fold'] == outer]

        for inner in np.unique(df.loc[df['fold'] != outer, 'fold']):

            val = proxy.loc[proxy['fold'] == inner]
            train = proxy.loc[(proxy['fold'] != outer) & (proxy['fold'] != inner)]
            yield deepcopy((train, val, test))






