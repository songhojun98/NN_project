import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd

def mol2_to_graph(mol2_file):
    # ligand의 .mol2 파일 읽기
    molecule = Chem.MolFromMol2File(mol2_file)
    if molecule is None:
        raise ValueError(f"Failed to load molecule from file: {mol2_file}")
    conf = molecule.GetConformer()

    # 원자와 결합을 나타내는 텐서 만들기
    atom_features = []
    atom_positions = []
    bond_indices = []
    bond_features = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
        atom_positions.append(conf.GetAtomPosition(atom.GetIdx()))

    for bond in molecule.GetBonds():
        bond_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bond_features.append(bond.GetBondTypeAsDouble())

    # PyTorch 텐서로 변환
    atom_features = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    atom_positions = torch.tensor(atom_positions, dtype=torch.float)
    bond_indices = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
    bond_features = torch.tensor(bond_features, dtype=torch.float).view(-1, 1)

    # PyTorch Geometric 데이터 객체 만들기
    data = Data(x=atom_features, pos=atom_positions, edge_index=bond_indices, edge_attr=bond_features)

    return data

def pdb_to_graph(pdb_file):
    # protein pocket .pdb 파일 읽기
    molecule = Chem.MolFromPDBFile(pdb_file)
    if molecule is None:
        raise ValueError(f"Failed to load molecule from file: {pdb_file}")
    conf = molecule.GetConformer()

    # 원자와 결합을 나타내는 텐서 만들기
    atom_features = []
    atom_positions = []
    bond_indices = []
    bond_features = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
        atom_positions.append(conf.GetAtomPosition(atom.GetIdx()))

    for bond in molecule.GetBonds():
        bond_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bond_features.append(bond.GetBondTypeAsDouble())

    # PyTorch 텐서로 변환
    atom_features = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    atom_positions = torch.tensor(atom_positions, dtype=torch.float)
    bond_indices = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
    bond_features = torch.tensor(bond_features, dtype=torch.float).view(-1, 1)

    # PyTorch Geometric 데이터 객체 만들기
    data = Data(x=atom_features, pos=atom_positions, edge_index=bond_indices, edge_attr=bond_features)

    return data


# binding affinity 데이터 불러오기
affinity_df = pd.read_csv("./INDEX_refined_data.2020", sep="\s+", header=None, names=["PDB", "resolution", "release_year", "Affinity"])

# 각 단백질-리간드 복합체와 일치하는 binding affinity 매칭
affinity_dict = pd.Series(affinity_df.Affinity.values, index=affinity_df.PDB).to_dict()

# 데이터셋의 경로
dataset_path = "./refined-set"

# 모든 ligand 및 protein_pocket 데이터를 저장할 리스트
ligands = []
proteins = []

# 모든 ligand .mol2 및 protein_pocket .pdb 파일에 대해
for root, dirs, files in os.walk(dataset_path):
    mol2_files = [file for file in files if file.endswith(".mol2")]
    pdb_files = [file for file in files if file.endswith("pocket.pdb")]
    
    for mol2_file, pdb_file in zip(mol2_files, pdb_files):
        pdb_code = os.path.basename(root)  # 디렉토리 이름을 따와서 PDB code 만들기

        mol2_file = os.path.join(root, mol2_file)
        pdb_file = os.path.join(root, pdb_file)

        try:
            ligand_data = mol2_to_graph(mol2_file)
            protein_data = pdb_to_graph(pdb_file)

            # PDB code와 일치하는 binding affinity 값 매칭시켜주기
            ligand_data.affinity = affinity_dict.get(pdb_code, "None")
            protein_data.affinity = affinity_dict.get(pdb_code, "None")

            # 리스트에 추가
            ligands.append(ligand_data)
            proteins.append(protein_data)

        except ValueError as e:
            print(f"Failed to process files {mol2_file} and {pdb_file}: {e}")
            continue  # 파일에 오류가 있을 시 단백질, 리간드 쌍에 대한 데이터 날리기
