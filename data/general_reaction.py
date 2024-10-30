import argparse
import pandas as pd
import torch
from torch_geometric.data import  Data
import numpy as np 
from rdkit import Chem
import os
from tqdm import tqdm
from molvs import standardize_smiles
import sys
from data.datasets import reaction_graph
from sklearn.model_selection import StratifiedKFold, KFold

from icecream import ic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class reaction_representation(reaction_graph):

    def __init__(self, opt:argparse.Namespace, filename: str, molcols: list, root: str = None, include_fold = True) -> None:

        self._include_fold = include_fold

        if self._include_fold:
            try:
                file_folds = filename[:-4] + '_folds' + filename[-4:]
                pd.read_csv(os.path.join(root, 'raw', f'{file_folds}'))
            except:
                self.split_data(root, filename, opt.folds, opt.global_seed)
            filename = filename[:-4] + '_folds' + filename[-4:]

        super().__init__(opt = opt, filename = filename, mol_cols = molcols, root=root)

        self._name = "general_reaction"

        
    def process(self):

        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        self.uni_vals = {}

        for feat in self._opt.ohe_graph_feat:
            self.uni_vals[feat] = self.data[feat].unique()

        for index, reaction in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            node_feats_reaction = None

            smiles_components = []

            for reactant in self.mol_cols:  

                #create a molecule object from the smiles string
                smiles = standardize_smiles(reaction[reactant])
                smiles_components.append(smiles)

                mol = Chem.MolFromSmiles(smiles)

                mol = Chem.rdmolops.AddHs(mol)

                #initialize the graph level features
                global_features = []

                # checks the global features that the user wants to include
                for global_feat in self._opt.graph_features.keys():

                    # checks if the molecule has the global feature
                    if reactant in self._opt.graph_features[global_feat] or reactant == 'all':

                        # checks if the global feature has to be one hot encoded
                        if global_feat in self.uni_vals.keys():
                            #get the unique values of the global feature
                            global_features += self._one_h_e(reaction[global_feat], self.uni_vals[global_feat])
                
                        else:
                            global_features += [reaction[global_feat]]
                    
                    # condition where the molecule does not have the global feature but it is necessary to add the '0'
                    else:
                        if global_feat in self.uni_vals.keys():
                            global_features += self._one_h_e('qWeRtYuIoP', self.uni_vals[global_feat], 'qWeRtYuIoP')
                        else:
                            global_features += [0]


                node_feats = self._get_node_feats(mol, global_features)

                node_feats = self._get_node_feats(mol)

                edge_attr, edge_index = self._get_edge_features(mol)

                if node_feats_reaction is None:
                    node_feats_reaction = node_feats
                    edge_index_reaction = edge_index
                    edge_attr_reaction = edge_attr

                else:
                    node_feats_reaction = torch.cat([node_feats_reaction, node_feats], axis=0)
                    edge_attr_reaction = torch.cat([edge_attr_reaction, edge_attr], axis=0)
                    edge_index += max(edge_index_reaction[0]) + 1
                    edge_index_reaction = torch.cat([edge_index_reaction, edge_index], axis=1)

            y = torch.tensor(reaction['ddG']).reshape(1)

            if self._include_fold:
                fold = reaction['fold']
            else:
                fold = None

            data = Data(x=node_feats_reaction, 
                        edge_index=edge_index_reaction, 
                        edge_attr=edge_attr_reaction, 
                        y=y,
                        smiles = smiles_components,
                        idx = index,
                        fold = fold
                        ) 
            
            torch.save(data, 
                       os.path.join(self.processed_dir, 
                                    f'reaction_{index}.pt'))
    

    def _get_node_feats(self, mol, graph_feats = None):

        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats += self._one_h_e(atom.GetSymbol(), self._elem_list)
            # Feature 2: Atom degree
            node_feats += self._one_h_e(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
            # Feature 3: Formal Charge
            node_feats += self._one_h_e(atom.GetFormalCharge(), [-1, 0, 1])
            # Feature 4: Chirality
            node_feats += self._one_h_e(atom.GetChiralTag(), [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
                                                              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
                                                              ],
                                                              Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            # Feature 5: Num Hs
            node_feats += self._one_h_e(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
            # Feature 6: Hybridization
            node_feats += self._one_h_e(atom.GetHybridization(), [Chem.rdchem.HybridizationType.S,
                                                                  Chem.rdchem.HybridizationType.SP,
                                                                  Chem.rdchem.HybridizationType.SP2,
                                                                  Chem.rdchem.HybridizationType.SP3,
                                                                  Chem.rdchem.HybridizationType.SP3D,
                                                                  Chem.rdchem.HybridizationType.SP3D2],
                                                                  Chem.rdchem.HybridizationType.UNSPECIFIED,)
            # Feature 7: Aromaticity
            node_feats += [atom.GetIsAromatic()]
            # Feature 8: In Ring
            node_feats += [atom.IsInRing()]

            if graph_feats is not None:
                node_feats += graph_feats

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats, dtype=np.float32)
        return torch.tensor(all_node_feats, dtype=torch.float)
    

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float)
    
    
    def _get_edge_features(self, mol):

        all_edge_feats = []
        edge_indices = []

        for bond in mol.GetBonds():

            #list to save the edge features
            edge_feats = []

            # Feature 1: Bond type (as double)
            edge_feats += self._one_h_e(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])

            #feature 2: double bond stereochemistry
            edge_feats += self._one_h_e(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE], Chem.rdchem.BondStereo.STEREONONE)

            # Feature 3: Is in ring
            edge_feats.append(bond.IsInRing())

            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

            # Append edge indices to list (twice, per direction)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            #create adjacency list
            edge_indices += [[i, j], [j, i]]

        all_edge_feats = np.asarray(all_edge_feats)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return torch.tensor(all_edge_feats, dtype=torch.float), edge_indices
    

    def _create_folds(num_folds, df):
        """
        splits a dataset in a given quantity of folds

        Args:
        num_folds = number of folds to create
        df = dataframe to be splited

        Returns:
        dataset with new "folds" and "mini_folds" column with information of fold for each datapoint
        """

        # Calculate the number of data points in each fold
        fold_size = len(df) // num_folds
        remainder = len(df) % num_folds

        # Create a 'fold' column to store fold assignments
        fold_column = []

        # Assign folds
        for fold in range(1, num_folds + 1):
            fold_count = fold_size
            if fold <= remainder:
                fold_count += 1
            fold_column.extend([fold] * fold_count)

        # Assign the 'fold' column to the DataFrame
        df['fold'] = fold_column

        return df
    

    def split_data(self, root, filename, n_folds, random_seed):

        dataset = pd.read_csv(os.path.join(root, 'raw', f'{filename}'))


        folds = KFold(n_splits = n_folds, shuffle = True, random_state=random_seed)

        test_idx = []

        for _, test in folds.split(np.zeros(len(dataset)), dataset['ddG']):
            test_idx.append(test)

        index_dict = {index: list_num for list_num, index_list in enumerate(test_idx) for index in index_list}

        dataset['fold'] = dataset.index.map(index_dict)

        filename = filename[:-4] + '_folds' + filename[-4:]

        dataset.to_csv(os.path.join(root, 'raw', filename))

        print('{}.csv file was saved in {}'.format(filename, os.path.join(root, 'raw')))

