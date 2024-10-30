import sys
import os
import torch
import time
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


from data.general_reaction import reaction_representation
from data.rhcaa import rhcaa_diene
from options.base_options import BaseOptions
from call_methods import make_network, create_loaders
from utils.utils_model import train_network, eval_network, network_report, network_outer_report
from icecream import ic

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def train_network_nested_cv(opt) -> None:

    print('Initialising chiral diene ligands experiment using early stopping')


    # Set the device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    data = rhcaa_diene(opt, opt.filename, opt.mol_cols, root=opt.root)


    # Create the loaders and nested cross validation iterators
    ncv_iterators = create_loaders(data, opt)

    # Initiate the counter of the total runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)

    # Loop through the nested cross validation iterators
    # The outer loop is for the outer fold or test fold
    for outer in range(1, opt.folds+1):
        # The inner loop is for the inner fold or validation fold
        for inner in range(1, opt.folds):

            # Inner fold is incremented by 1 to avoid having same inner and outer fold number for logging purposes
            real_inner = inner +1 if outer <= inner else inner

            # Initiate the early stopping parameters
            val_best_loss = 1000
            early_stopping_counter = 0
            best_epoch = 0
            # Increment the counter
            counter += 1
            # Get the data loaders
            train_loader, val_loader, test_loader = next(ncv_iterators)
            # Initiate the lists to store the losses
            train_list, val_list, test_list = [], [], []
            # Create the GNN model
            model = make_network(network_name = "GCN",
                                 opt = opt, 
                                 n_node_features= data.num_node_features).to(device)
            
            # Start the timer for the training
            start_time = time.time()

            for epoch in range(opt.epochs):
                # Checks if the early stopping counter is less than the early stopping parameter
                if early_stopping_counter <= opt.early_stopping:
                    # Train the model
                    train_loss = train_network(model, train_loader, device)
                    # Evaluate the model
                    val_loss = eval_network(model, val_loader, device)
                    test_loss = eval_network(model, test_loader, device)  

                    print('{}/{}-Epoch {:03d} | Train loss: {:.3f} {} | Validation loss: {:.3f} {} | '             
                        'Test loss: {:.3f} {}'.format(counter, TOT_RUNS, epoch, train_loss, opt.target_var_units, val_loss, opt.target_var_units, test_loss, opt.target_var_units,))
                    
                    # Model performance is evaluated every 5 epochs
                    if epoch % 5 == 0:
                        # Scheduler step
                        model.scheduler.step(val_loss)
                        # Append the losses to the lists
                        train_list.append(train_loss)
                        val_list.append(val_loss)
                        test_list.append(test_loss)
                        
                        # Save the model if the validation loss is the best
                        if val_loss < val_best_loss:
                            # Best validation loss and early stopping counter updated
                            val_best_loss, best_epoch = val_loss, epoch
                            early_stopping_counter = 0
                            print('New best validation loss: {:.4f} found at epoch {}'.format(val_best_loss, best_epoch))
                            # Save the  best model parameters
                            best_model_params = deepcopy(model.state_dict())
                        else:
                            # Early stopping counter is incremented
                            early_stopping_counter += 1

                    if epoch == opt.epochs:
                        print('Maximum number of epochs reached')

                else:
                    print('Early stopping limit reached')
                    break
            
            print('---------------------------------')
            # End the timer for the training
            training_time = (time.time() - start_time)/60
            print('Training time: {:.2f} minutes'.format(training_time))

            print(f"Training for test outer fold: {outer}, and validation inner fold: {real_inner} completed.")
            print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")

            print('---------------------------------')

            # Report the model performance
            network_report(
                log_dir=f"{opt.log_dir_results}/{opt.filename[:-4]}/learning/results_GNN/",
                loaders=(train_loader, val_loader, test_loader),
                outer=outer,
                inner=real_inner,
                loss_lists=(train_list, val_list, test_list),
                save_all=True,
                model=model,
                model_params=best_model_params,
                best_epoch=best_epoch,
                opt=opt,
            )

            # Reset the variables of the training
            del model, train_loader, val_loader, test_loader, train_list, val_list, test_list, best_model_params, best_epoch
        
        print(f'All runs for outer test fold {outer} completed')
        print('Generating outer report')

        network_outer_report(
            log_dir=f"{opt.log_dir_results}/{opt.filename[:-4]}/learning/results_GNN/Fold_{outer}_test_set/",
            outer=outer,
            folds=opt.folds,
        )

        print('---------------------------------')
    
    print('All runs completed')
    

if __name__ == "__main__":
    opt = BaseOptions().parse() 
    train_network_nested_cv(opt)

