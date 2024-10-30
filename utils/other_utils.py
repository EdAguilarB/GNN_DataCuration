import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import AddHs
from torch_geometric.loader import DataLoader
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px


from icecream import ic

######################################
######################################
############XAI FUNCTIONS#############
######################################
######################################

def explain_dataset(dataset: List, 
                   explainer,
                   mol: Optional[str] = None,):
    
    #Creates a loader object from the dataset
    loader = DataLoader(dataset=dataset)

    # list to store the masks
    masks_ligand = []
    masks_substrate = []
    masks_boron = []
    all_masks = []

    #Iterate over the graphs in the loader
    for graph in tqdm(loader):

        # Get the number of atoms in each molecule
        na_ligand = AddHs(Chem.MolFromSmiles(graph.ligand[0])).GetNumAtoms()
        na_substrate = AddHs(Chem.MolFromSmiles(graph.substrate[0])).GetNumAtoms()
        na_boron = AddHs(Chem.MolFromSmiles(graph.boron[0])).GetNumAtoms()

        # Define the index of the first and last atom of each molecule
        ia_ligand = 0
        fa_ligand = na_ligand

        ia_substrate = na_ligand
        fa_substrate = na_ligand + na_substrate

        ia_boron =  na_ligand + na_substrate
        fa_boron = na_ligand + na_substrate + na_boron

        ia = 0
        fa = na_ligand+na_substrate+na_boron

        # Run the explanation function over the reaction graph
        explanation = explainer(x = graph.x, 
                                edge_index=graph.edge_index,  
                                batch_index=graph.batch)
        
        # Get the masks for each node within the molecule
        masks = explanation.node_mask

        masks = masks/torch.max(masks.sum(dim=1))


        # Append the masks for each molecule to the list
        masks_ligand.append(masks[ia_ligand:fa_ligand])
        masks_substrate.append(masks[ia_substrate: fa_substrate])
        masks_boron.append(masks[ia_boron: fa_boron])

        all_masks.append(masks[ia:fa])

    # Concatenate the masks for each molecule
    masks_ligand = torch.cat(masks_ligand, dim = 0)
    masks_substrate = torch.cat(masks_substrate, dim = 0)
    masks_boron = torch.cat(masks_boron, dim = 0)
    all_masks = torch.cat(all_masks, dim=0)

    if mol == 'ligand':
        return masks_ligand
    elif mol == 'substrate':
        return masks_substrate
    elif mol == 'boron':
        return masks_boron
    else:
        return masks_ligand, masks_substrate, masks_boron, all_masks

def plot_denoised_mols(mask,
                       graph,
                       mol: str,
                       analysis:str=None,
                       norm:bool=True,):
    
    atom_identity = 10
    degree = 4
    hyb = 4
    aromatic = 1
    ring = 1
    chiral = 2
    conf = 2

    importances = []
    importances.append(mask[:, 0:atom_identity])
    importances.append(mask[:, atom_identity:atom_identity+degree])
    importances.append(mask[:, atom_identity+degree:atom_identity+degree+hyb])
    importances.append(mask[:, atom_identity+degree+hyb:atom_identity+degree+hyb+aromatic])
    importances.append(mask[:, atom_identity+degree+hyb+aromatic:atom_identity+degree+hyb+aromatic+ring])
    importances.append(mask[:, atom_identity+degree+hyb+aromatic+ring:atom_identity+degree+hyb+aromatic+ring+chiral])
    importances.append(mask[:, atom_identity+degree+hyb+aromatic+ring+chiral:atom_identity+degree+hyb+aromatic+ring+chiral+conf])

    if analysis:
        if analysis == 'atom_identity':
            importance = importances[0]
            importance = importance.sum(dim=1).cpu().numpy()

        elif analysis == 'degree':
            importance = importances[1]
            importance = importance.sum(dim=1).cpu().numpy()

        elif analysis == 'hyb':
            importance = importances[2]
            importance = importance.sum(dim=1).cpu().numpy()

        elif analysis == 'aromatic':
            importance = importances[3]
            importance = importance.sum(dim=1).cpu().numpy()

        elif analysis == 'ring':
            importance = importances[4]
            importance = importance.sum(dim=1).cpu().numpy()

        elif analysis == 'chiral':
            importance = importances[5]
            importance = importance.sum(dim=1).cpu().numpy()

        elif analysis == 'conf':
            importance = importances[6]
            importance = importance.sum(dim=1).cpu().numpy()

        else:
            importance = mask.sum(dim=1).cpu().numpy()

    else:
        importance = mask.sum(dim=1).cpu().numpy()

    if mol == 'ligand':
        smiles = graph.ligand[0]

    elif mol == 'substrate':
        smiles = graph.substrate[0]
    
    elif mol == 'boron':
        smiles = graph.boron[0]
    
    plot_weighted_mol(importance, 
                      smiles,
                      norm)

def plot_weighted_mol(mask, 
                      smiles:str,
                      norm=False
                      ):

    mol = AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)

    atoms = mol.GetNumAtoms()
    coords = mol.GetConformer().GetPositions()
    atom_symbol = [atom.GetSymbol() for atom in mol.GetAtoms()]

    edge_idx = []

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_idx += [[u, v], [v, u]]

    edge_idx = np.array(edge_idx).T
    edge_coords = dict(zip(range(atoms), coords))

    coords_edges = [(np.concatenate([np.expand_dims(edge_coords[u], axis=1), np.expand_dims(edge_coords[v], axis =1)], 
                                axis = 1)) for u, v in zip(edge_idx[0], edge_idx[1])]
    
    ic(mask)

    if norm==True:
        mask = mask/np.max(mask)

    ic(mask)

    mask = np.where(mask < 0.6, np.power(mask, 2), np.sqrt(mask))

    ic(mask)


    atoms_trace = trace_atoms(atom_symbol = atom_symbol,
                              coords=coords,
                              sizes=sizes,
                              colors=colors_n,
                              transparencies=mask)
    
    edges_trace = trace_bonds(coords_edges=coords_edges, 
                              edge_mask_dict=edge_idx[0])
    
    traces = atoms_trace + edges_trace

    fig = go.Figure(data=traces)
    fig.update_layout(template =  'plotly_white')
    fig.update_layout(scene=dict(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                xaxis_title='', yaxis_title='', zaxis_title=''))
    fig.show()


def visualize_score_features(
    score: torch.Tensor,
    graph = None,
    analysis: Optional[str] = None, 
    mol: Optional[str] = None,
):
    
    '''
    Function to visualize the importance of the node features
    in the graph. The function can be used to visualize the
    importance of the node features in the ligand, substrate,
    boron or the entire graph. The function will resturn a plot
    showing the importance of the node features given the
    masks calculated by any method
    '''

    atom_identity = 10
    degree = 4
    hyb = 4
    aromatic = 1
    ring = 1
    chiral = 2
    conf = 2
    temp = 1


    if mol:

        na_ligand = AddHs(Chem.MolFromSmiles(graph.ligand[0])).GetNumAtoms()
        na_substrate = AddHs(Chem.MolFromSmiles(graph.substrate[0])).GetNumAtoms()
        na_boron = AddHs(Chem.MolFromSmiles(graph.boron[0])).GetNumAtoms()

        if mol == 'ligand':
            ia = 0
            fa = na_ligand
        elif mol == 'substrate':
            ia = na_ligand
            fa = na_ligand + na_substrate
        elif mol == 'boron':
            ia =  na_ligand + na_substrate
            fa =  na_ligand + na_substrate + na_boron

        importances = []
        importances.append(score[ia:fa, 0:atom_identity])
        importances.append(score[ia:fa, atom_identity:atom_identity+degree])
        importances.append(score[ia:fa, atom_identity+degree:atom_identity+degree+hyb])
        importances.append(score[ia:fa, atom_identity+degree+hyb:atom_identity+degree+hyb+aromatic])
        importances.append(score[ia:fa, atom_identity+degree+hyb+aromatic:atom_identity+degree+hyb+aromatic+ring])
        importances.append(score[ia:fa, atom_identity+degree+hyb+aromatic+ring:atom_identity+degree+hyb+aromatic+ring+chiral])
        importances.append(score[ia:fa, atom_identity+degree+hyb+aromatic+ring+chiral:atom_identity+degree+hyb+aromatic+ring+chiral+conf])
        importances.append(score[ia:fa, atom_identity+degree+hyb+aromatic+ring+chiral+conf:atom_identity+degree+hyb+aromatic+ring+chiral+conf+temp])
    
    else:

        importances = []
        importances.append(score[:, 0:atom_identity])
        importances.append(score[:, atom_identity:atom_identity+degree])
        importances.append(score[:, atom_identity+degree:atom_identity+degree+hyb])
        importances.append(score[:, atom_identity+degree+hyb:atom_identity+degree+hyb+aromatic])
        importances.append(score[:, atom_identity+degree+hyb+aromatic:atom_identity+degree+hyb+aromatic+ring])
        importances.append(score[:, atom_identity+degree+hyb+aromatic+ring:atom_identity+degree+hyb+aromatic+ring+chiral])
        importances.append(score[:, atom_identity+degree+hyb+aromatic+ring+chiral:atom_identity+degree+hyb+aromatic+ring+chiral+conf])
        #importances.append(score[:, atom_identity+degree+hyb+aromatic+ring+chiral+conf:atom_identity+degree+hyb+aromatic+ring+chiral+conf+temp])

    if analysis:
        if analysis == 'atom_identity':
            importance = importances[0]
            importance = importance.sum(dim=0).cpu().numpy()
            labels = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'S', 'Cl', 'Br']
            title = 'Atom Identity Importance'

        elif analysis == 'degree':
            importance = importances[1]
            importance = importance.sum(dim=0).cpu().numpy()
            labels = [1, 2, 3, 4]
            title = 'Atom Degree Importance'

        elif analysis == 'hyb':
            importance = importances[2]
            importance = importance.sum(dim=0).cpu().numpy()
            labels = [0,2,3,4]
            title = 'Atom Hybridization Importance'

        elif analysis == 'aromatic':
            importance = importances[3]
            importance = importance.sum(dim=0).cpu().numpy()
            labels = ['aromatic']
            title = 'Atom Aromaticity Importance'

        elif analysis == 'ring':
            importance = importances[4]
            importance = importance.sum(dim=0).cpu().numpy()
            labels = ['IsInRing']
            title = 'Atoms in Ring Importance'

        elif analysis == 'chiral':
            importance = importances[5]
            importance = importance.sum(dim=0).cpu().numpy()
            labels = [0,1,2]
            title = 'Atom Chirality Importance'

        elif analysis == 'conf':
            importance = importances[6]
            importance = importance.sum(dim=0).cpu().numpy()
            labels = [2, 1]
            title = 'Ligand Configuration Importance'

    else:
        importances = [importance.sum().cpu().numpy() for importance in importances]
        importances = [np.array([x.item()]) for x in importances]
        importance = np.concatenate(importances)
        labels = ['Atom Identity', 'Atom Degree', 'Atom Hybridization', 'Atom Aromaticity', 
                  'Atom InRing', 'Atom Chirality', 'Ligand Configuration']
        title = 'Global Importance of Node Features'

    df = pd.DataFrame({'score': importance, 'labels': labels}, index=labels)
    df = df.sort_values('score', ascending=False)
    df = df.round(decimals=3)

    return df


def get_graph_by_idx(loader, idx):
    # Iterate over the loader to find the graph with the desired idx
    for data in loader:
        graph_idx = data.idx  # Access the idx attribute of the graph

        if graph_idx == idx:
            # Found the graph with the desired idx
            return data

    # If the desired graph is not found, return None or raise an exception
    return None


def mol_prep(mol_graph, mol:str):

    mol_l = AddHs(Chem.MolFromSmiles(mol_graph.ligand[0]))
    mol_s = AddHs(Chem.MolFromSmiles(mol_graph.substrate[0]))
    mol_b = AddHs(Chem.MolFromSmiles(mol_graph.boron[0]))

    atoms_l = mol_l.GetNumAtoms()
    atoms_s = mol_s.GetNumAtoms()
    atoms_b = mol_b.GetNumAtoms()

    if mol == 'l':
        fa = 0
        la = atoms_l

        AllChem.EmbedMolecule(mol_l, AllChem.ETKDGv3())
        coords = mol_l.GetConformer().GetPositions()
        atom_symbol = [atom.GetSymbol() for atom in mol_l.GetAtoms()]

    elif mol == 's':
        fa = atoms_l
        la = atoms_l + atoms_s 

        AllChem.EmbedMolecule(mol_s, AllChem.ETKDGv3())
        coords = mol_s.GetConformer().GetPositions()
        atom_symbol = [atom.GetSymbol() for atom in mol_s.GetAtoms()]

    elif mol == 'b':
        fa = atoms_l + atoms_s 
        la = atoms_l + atoms_s +atoms_b

        AllChem.EmbedMolecule(mol_b, AllChem.ETKDGv3())
        coords = mol_b.GetConformer().GetPositions()
        atom_symbol = [atom.GetSymbol() for atom in mol_b.GetAtoms()]

    else:
        print('No valid molecule selected')

    return fa, la, coords, atom_symbol


def get_masks(explanation, fa, la, edge_idx):
    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask

    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *edge_idx)):
        u, v = u.item(), v.item()
        if u in range(fa, la):
                if u > v:
                        u, v = v, u
                edge_mask_dict[(u, v)] += val.item()
    
    node_mask=node_mask[fa:la]

    return edge_mask_dict, node_mask

def normalise_masks(edge_mask_dict, node_mask):
    neg_edge = [True if num < 0 else False for num in list(edge_mask_dict.values())]
    min_value_edge = abs(min(edge_mask_dict.values(), key=abs))
    max_value_edge = abs(max(edge_mask_dict.values(), key=abs))

    abs_dict = {key: abs(value) for key, value in edge_mask_dict.items()}
    abs_dict = {key: (value - min_value_edge) / (max_value_edge - min_value_edge) 
                for key, value in abs_dict.items()}
    
    edge_mask_dict_norm = {key: -value if convert else value for (key, value), convert 
                      in zip(abs_dict.items(), neg_edge)}
    
    node_mask = node_mask.sum(axis = 1)
    node_mask = [val.item() for val in node_mask]
    neg_nodes = [True if num < 0 else False for num in node_mask]
    max_node = abs(max(node_mask, key = abs))
    min_node = abs(min(node_mask, key = abs))
    abs_node = [abs(w) for w in node_mask]
    abs_node = [(w-min_node)/(max_node-min_node) for w in abs_node]
    node_mask_norm = [-w if neg_nodes else w for w, neg_nodes in zip(abs_node, neg_nodes)]

    return edge_mask_dict_norm, node_mask_norm


colors_n = {
    'C': 'black',
    'O': 'red',
    'N': 'blue',
    'H': 'lightgray',
    'B': 'brown',
    'F': 'pink'
}

colors_cb = {
    'C': '#333333',
    'O': '#FF0000',
    'N': '#0000FF',
    'H': '#FFFFFF',
    'B': '#FFA500'
}

sizes = {
    'C': 69/2,
    'O': 66/2,
    'N': 71/2,
    'H': 31/2,
    'B': 84/2,
    'F': 64/2
}

def select_palette(palette, neg_nodes, neg_edges):

    if palette=='normal':
        colors = colors_n
        color_nodes = ['red' if boolean else 'blue' for boolean in neg_nodes]
        color_edges = ['red' if boolean else 'blue' for boolean in neg_edges]

    elif palette=='cb':
        colors = colors_cb
        color_nodes = ['#006400' if boolean else '#4B0082' for boolean in neg_nodes]
        color_edges = ['#006400' if boolean else '#4B0082' for boolean in neg_edges]
    
    return colors, color_nodes, color_edges

def trace_atoms(atom_symbol, coords, sizes, colors, transparencies=None):
    trace_atoms = [None] * len(atom_symbol)
    for i in range(len(atom_symbol)):
        marker_dict = {
            'symbol': 'circle',
            'size': sizes[atom_symbol[i]],
            'color': colors[atom_symbol[i]]
        }
        
        if transparencies is not None:
            marker_dict['opacity'] = transparencies[i]
        
        trace_atoms[i] = go.Scatter3d(
            x=[coords[i][0]],
            y=[coords[i][1]],
            z=[coords[i][2]],
            mode='markers',
            text=f'atom {atom_symbol[i]}',
            legendgroup='Atoms',
            showlegend=False,
            marker=marker_dict
        )
    return trace_atoms


def trace_atom_imp(coords, opacity, atom_symbol, sizes, color):
    trace_atoms_imp = [None] * len(atom_symbol)
    for i in range(len(atom_symbol)):

        trace_atoms_imp[i] = go.Scatter3d(x=[coords[i][0]],
                            y=[coords[i][1]],
                            z=[coords[i][2]],
                            mode='markers',
                            showlegend=False,
                            opacity=opacity[i],
                            text = f'atom {atom_symbol[i]}',
                            legendgroup='Atom importance',
                            marker=dict(symbol='circle',
                                                    size=sizes[atom_symbol[i]]*1.7,
                                                    color=color[i])
        )
    return trace_atoms_imp


def trace_bonds(coords_edges, edge_mask_dict):
    trace_edges = [None] * len(edge_mask_dict)
    
    for i in range(len(edge_mask_dict)):
        trace_edges[i]= go.Scatter3d(
            x=coords_edges[i][0],
            y=coords_edges[i][1],
            z=coords_edges[i][2],
            mode='lines',
            showlegend=False,
            legendgroup='Bonds',
            line=dict(color='black', width=2),
            hoverinfo='none')
        
    return trace_edges

def trace_bond_imp(coords_edges, edge_mask_dict, opacity, color_edges):
    trace_edge_imp = [None] * len(edge_mask_dict)
    for i in range(len(edge_mask_dict)):
        trace_edge_imp[i]= go.Scatter3d(
            x=coords_edges[i][0],
            y=coords_edges[i][1],
            z=coords_edges[i][2],
            mode='lines',
            showlegend=False,
            legendgroup='Bond importance',
            opacity=opacity[i],
            line=dict(color=color_edges[i], width=opacity[i]*15),
            hoverinfo='none')
        
    return trace_edge_imp


def all_traces(atoms, atoms_imp, bonds, bonds_imp):
    traces =   atoms + atoms_imp + bonds + bonds_imp
    fig = go.Figure(data=traces)
    fig.add_trace(go.Scatter3d(
    x=[None],
    y=[None],
    z=[None],
    mode='markers',
    legendgroup='Atoms',
    name='Atoms'
        ))

    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        legendgroup='Atom importance',
        name='Atom importance'
    ))

    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        legendgroup='Bonds',
        name='Bonds'
    ))

    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        legendgroup='Bond importance',
        name='Bond importance',
        showlegend=True
    ))

    fig.update_layout(template =  'plotly_white')

    fig.update_layout(scene=dict(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                xaxis_title='', yaxis_title='', zaxis_title=''))


    fig.show()

def plot_molecule_importance(mol_graph, mol, explanation, palette):

    edge_idx = mol_graph.edge_index
    fa, la, coords, atom_symbol = mol_prep(mol_graph=mol_graph, mol=mol)
    edge_coords = dict(zip(range(fa, la), coords))
    edge_mask_dict, node_mask = get_masks(explanation=explanation, fa=fa, la=la, edge_idx=edge_idx)

    edge_mask_dict, node_mask = normalise_masks(edge_mask_dict=edge_mask_dict, node_mask=node_mask)
    

    coords_edges = [(np.concatenate([np.expand_dims(edge_coords[u], axis=1), np.expand_dims(edge_coords[v], axis =1)], 
                                axis = 1)) for u, v in edge_mask_dict.keys()]
    
    edge_weights = list(edge_mask_dict.values())
    opacity_edges = [(x + 1) / 2 for x in edge_weights]
    opacity_nodes = [(x + 1) / 2 for x in node_mask]

    neg_edges = [True if num < 0 else False for num in list(edge_mask_dict.values())]
    neg_nodes = [True if num < 0 else False for num in node_mask]

    colors_atoms, color_nodes_imp, color_edges_imp = select_palette(palette=palette, neg_nodes=neg_nodes, neg_edges=neg_edges)

    atoms = trace_atoms(atom_symbol = atom_symbol, coords=coords,sizes=sizes, 
                        colors=colors_atoms)
    atoms_imp = trace_atom_imp(coords=coords, opacity=opacity_nodes, 
                               atom_symbol=atom_symbol, sizes=sizes,color=color_nodes_imp)
    bonds = trace_bonds(coords_edges=coords_edges, edge_mask_dict=edge_mask_dict)
    bond_imp = trace_bond_imp(coords_edges=coords_edges, edge_mask_dict=edge_mask_dict,
                              opacity=opacity_edges, color_edges=color_edges_imp)
    
    all_traces(atoms=atoms, atoms_imp=atoms_imp, bonds=bonds, bonds_imp=bond_imp)
