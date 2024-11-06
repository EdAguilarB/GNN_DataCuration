import argparse
import os


class BaseOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        ###########################################
        ########Options to run experiments#########
        ###########################################

        self.parser.add_argument(
            '--train_GNN', 
            type=self.str2bool,
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to train the GNN or not'
            )
        
        
        self.parser.add_argument(
            '--experiment_name',
            type=str,
            default='experiment',
            help='name of the experiment',
            ),
        

        ###########################################
        ##############Options Dataset##############
        ###########################################

        self.parser.add_argument(
            '--root', 
            type=str, 
            default='data/datasets/biaryl/',
            help='path to the folder containing the csv files',
            )
        
        self.parser.add_argument(
            '--filename',
            type=str,
            default='biaryl.csv',
            help='name of the csv file',
            )
        
        self.parser.add_argument(
            '--unseen_ratio',
            type=float,
            default=0.2,
            help='ratio of unseen data',
            )
        
        
        
        ###########################################
        ##########Options to log results###########
        ###########################################
        self.parser.add_argument(
            '--log_dir_results',
            type=str,
            default=os.path.join('results/'),
            help='path to the folder where the results will be saved',
            )
        
        ###########################################
        #########Smiles columns in dataset#########
        ###########################################

        self.parser.add_argument(
            '--mol_cols',
            type=str,
            default=['ligand', 'substrate', 'boronreagent'],
            help='column names of the reactant and product smiles',
            )
        
        self.parser.add_argument(
            '--target_var',
            type=str,
            default='%top',
            help='Whether to include the fold column in the dataset',
            )

        self.parser.add_argument(
            '--target_var_units',
            type=str,
            default='%',
            help='Whether to include the fold column in the dataset',
            )
        
        self.parser.add_argument(
            '--graph_features',
            type=dict,
            default={'Confg': ['ligand'],
                     'Temp': ['all'],
                     }
                     ,
            help='Global features to include in the graph',
            )
        
        self.parser.add_argument(
            '--ohe_graph_feat',
            type=dict,
            default=['Confg'],
            help='Number of node features',
            )

        
        ###########################################
        ############Training Options GNN###########
        ###########################################
        
        self.parser.add_argument(
            '--folds',
            type=int,
            default=10,
            help='Number of folds',
            )
        
        self.parser.add_argument(
            '--n_classes',
            type=int,
            default=1,
            help='Number of classes',
            )
        
        self.parser.add_argument(
            '--n_convolutions',
            type=int,
            default=2,
            help='Number of convolutions',
            )
        
        self.parser.add_argument(
            '--readout_layers',
            type=int,
            default=2,
            help='Number of readout layers',
            )
        
        self.parser.add_argument(
            '--embedding_dim',
            type=int,
            default=64,
            help='Embedding dimension',
            )
        
        self.parser.add_argument(
            '--improved',
            type=bool,
            default=True,
            help='Whether to use the improved version of the GCN',
            )
        
        self.parser.add_argument(
            '--problem_type',
            type=str,
            default='regression',
            help='Type of problem',
            )
        
        self.parser.add_argument(
            '--optimizer',
            type=str,
            default='Adam',
            help='Type of optimizer',
            )
        
        self.parser.add_argument(
            '--lr',
            type=float,
            default=0.01,
            help='Learning rate',
            )
        
        self.parser.add_argument(
            '--early_stopping',
            type=int,
            default=6,
            help='Early stopping',
            )
        
        self.parser.add_argument(
            '--scheduler',
            type=str,
            default='ReduceLROnPlateau',
            help='Type of scheduler',
            )
        
        self.parser.add_argument(
            '--step_size',
            type=int,
            default=7,
            help='Step size for the scheduler',
            )
        
        self.parser.add_argument(
            '--gamma',
            type=float,
            default=0.7,
            help='Factor for the scheduler',
            )
        
        self.parser.add_argument(
            '--min_lr',
            type=float,
            default=1e-08,
            help='Minimum learning rate for the scheduler',
            )
        
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=40,
            help='Batch size',
            )
        
        self.parser.add_argument(
            '--epochs',
            type=int,
            default=250,
            help='Number of epochs',
            )  

        
        self.parser.add_argument(
            '--global_seed',
            type=int,
            default=20232023,
            help='Global random seed for reproducibility',
            )
        
        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt
    
    @staticmethod
    def str2bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
