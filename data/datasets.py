import argparse
import os
import sys
import torch
import pandas as pd
from torch_geometric.data import Dataset

from icecream import ic


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class reaction_graph(Dataset):


    def __init__(self, opt: argparse.Namespace, filename: str, mol_cols: list, root: str) -> None:

        self.filename = filename
        self.mol_cols = mol_cols
        self._name = "BaseDataset"
        self._opt = opt
        self._root = root
        
        super().__init__(root = self._root)
        

    @property
    def raw_file_names(self):
        return self.filename
    
    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        molecules = [f'reaction_{i}.pt' for i in list(self.data.index)]
        return molecules
    
    @property
    def _elem_list(self):
        elements = [
            'H', 
            'B', 
            'C', 
            'N', 
            'O', 
            'F', 
            'Si', 
            'S', 
            'Cl', 
            'Br',
            'P',
            'K',
            'Li',
            'I',
            'Na']
        
        return elements
        
    
    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError
    
    def _get_node_feats(self):
        raise NotImplementedError
    
    def _get_edge_features(self):
        raise NotImplementedError
    
    def _print_dataset_info(self) -> None:
        """
        Prints the dataset info
        """
        print(f"{self._name} dataset has {len(self)} samples")

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):

        molecule = torch.load(os.path.join(self.processed_dir, 
                                f'reaction_{idx}.pt'), weights_only=False) 
        return molecule
    
    def _get_atom_chirality(self, CIP_dict, atom_idx):
        try:
            chirality = CIP_dict[atom_idx]
        except KeyError:
            chirality = 'No_Stereo_Center'

        return chirality
    
    def _one_h_e(self, x, allowable_set, ok_set=None):

        if x not in allowable_set:
            if ok_set is not None and x == ok_set:
                pass
            else:
                print(x)
        return list(map(lambda s: x == s, allowable_set))
    
