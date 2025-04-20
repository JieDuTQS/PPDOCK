import pandas as pd
import torch
import lmdb
import pickle

def process_data(txt_file_path, lmdb_file_path, pt_file_path, output_file_path):
    columns = ["protein_name", "compound_name", "pdb", "smiles", "affinity", "uid",
               "pocket_com", "use_compound_com", "use_whole_protein", "group", 
               "p_length", "c_length", "y_length", "num_contact", "native_num_contact"]
    df = pd.DataFrame(columns=columns)

    lmdb_env = lmdb.open(lmdb_file_path, readonly=True, lock=False)
    ligand_dict = torch.load(pt_file_path)
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data_name = line.strip()
        
        data = {
            "protein_name": data_name,
            "compound_name": data_name,
            "pdb": data_name,
            "smiles": None,
            "affinity": None,
            "uid": None,
            "pocket_com": None,
            "use_compound_com": True,
            "use_whole_protein": False,
            "group": "test",
            "p_length": None,
            "c_length": None,
            "y_length": None,
            "num_contact": None,
            "native_num_contact": None
        }

        with lmdb_env.begin() as txn:
            protein_data = txn.get(data_name.encode('utf-8'))
            if protein_data is not None:
                protein_tuple = pickle.loads(protein_data)
                protein_coords = protein_tuple[0]  
                data["p_length"] = protein_coords.size()[0]

        ligand_coords = ligand_dict.get(data_name)
        if ligand_coords is None:
            print('\n\nNo ligand_coords: ',data_name,'\n\n')
        if ligand_coords is not None:
            data["c_length"] = len(ligand_coords)
        df = df.append(data, ignore_index=True)
    torch.save(df, output_file_path)
    print(f"Saved to {output_file_path}")

txt_file_path = 'data_ori/list.txt'
lmdb_file_path = 'data_processed/protein_1d_3d.lmdb'
pt_file_path = 'data_processed/compound_rdkit_coords.pt'
output_file_path = 'data_processed/data.pt'

process_data(txt_file_path, lmdb_file_path, pt_file_path, output_file_path)