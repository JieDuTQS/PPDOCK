# 'protein_1d_3d.lmdb'、'esm2_t33_650M_UR50D.lmdb'
from Bio.PDB import PDBParser
import torch
from tqdm import tqdm
import os
import esm
import argparse
import lmdb
import pickle

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

def get_protein_structure(res_list):
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    return structure


def extract_protein_structure(path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", path) 
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    sturcture = get_protein_structure(res_list) 
    return sturcture



def extract_esm_feature(protein):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                    'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                    'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                    'N': 2, 'Y': 18, 'M': 12}

    num_to_letter = {v:k for k, v in letter_to_num.items()}

    # Load ESM-2 model with different sizes
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", protein['seq']),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_representations = results["representations"][33][0][1: -1]
    assert token_representations.shape[0] == len(protein['seq'])
    return token_representations

def map_string_to_numbers(string, index_dict):
    number_list = []

    for char in string:
        if char in index_dict:
            number_list.append(index_dict[char])
        else:
            raise ValueError(f"Character '{char}' not found in the index dictionary.")
    
    return number_list


def process_protein_data(txt_file_path, data_folder, output_folder,residue_to_num):
    lmdb_path = os.path.join(output_folder, 'protein_1d_3d.lmdb')
    env = lmdb.open(lmdb_path, map_size=int(1e12))  
    protein_esm2_db = lmdb.open(os.path.join(output_folder, 'esm2_t33_650M_UR50D.lmdb'), map_size=1024 ** 4)
    
    with open(txt_file_path, 'r') as f:
        data_names = f.read().splitlines()
    
    with env.begin(write=True) as txn,protein_esm2_db.begin(write=True) as txn1:
        for data_name in tqdm(data_names):
            subfolder_path = os.path.join(data_folder, data_name)
            if not os.path.isdir(subfolder_path):
                print(f"Warning: Subfolder {subfolder_path} not found for data {data_name}")
                continue

            
            pdb_file = None
            for root, dirs, files in os.walk(subfolder_path):
                for file in files:
                    if file.endswith('protein.pdb') and file.startswith(data_name):
                        pdb_file = os.path.join(root, file)
                        break
                if pdb_file:
                    break
            
            if not pdb_file:
                print(f"Warning: .pdb file not found in {subfolder_path}")
                continue
            
            
            protein_structure = extract_protein_structure(pdb_file)
            protein_structure['name'] = data_name

            esm2_feature = extract_esm_feature(protein_structure)

            ca_coordinates_tensor = torch.tensor(protein_structure['coords'], dtype=torch.float32)[:, 1]
            encoded_sequence=map_string_to_numbers(protein_structure['seq'],residue_to_num)
            encoded_sequence_tensor = torch.tensor(encoded_sequence, dtype=torch.long)

            protein_data = (ca_coordinates_tensor, encoded_sequence_tensor)

            txn.put(data_name.encode('utf-8'), pickle.dumps(protein_data))
            txn1.put(data_name.encode('utf-8'), pickle.dumps(esm2_feature))

    env.close()
    protein_esm2_db.close()
    print(f"Data processing complete. LMDB file stored at: {lmdb_path}")
    
residue_to_num ={'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                    'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                    'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                    'N': 2, 'Y': 18, 'M': 12}

process_protein_data('data_ori/list.txt', 'data_ori/test_data', 'data_processed',residue_to_num)