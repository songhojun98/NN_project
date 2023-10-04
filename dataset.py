import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, Batch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

class PLPairDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        self.processed_counter = 0
        super(PLPairDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        pdb_codes = [name for name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, name))]
        return pdb_codes

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(self.processed_counter)]

    def process(self):
        pdb_codes = self.raw_file_names
        for i in range(len(pdb_codes)):
            pdb_code = pdb_codes[i]

            ligand_file = os.path.join(self.root, pdb_code, f'{pdb_code}_ligand.pkl')
            protein_file = os.path.join(self.root, pdb_code, f'{pdb_code}_protein.pkl')

            if not os.path.isfile(ligand_file) or not os.path.isfile(protein_file):
                continue  # 파일이 존재하지 않으면 건너뛰기

            # Load data from each file
            with open(ligand_file, "rb") as lf:
                ligand_data = pickle.load(lf)
            with open(protein_file, "rb") as pf:
                protein_data = pickle.load(pf)

            # 단백질과 리간드를 결합하여 단일 데이터 객체 만들기
            data = Data(protein=protein_data, ligand=ligand_data)

            # 처리된 데이터를 디스크에 저장
            torch.save(data, os.path.join(self.processed_dir, f'data_{self.processed_counter}.pt'))

            self.processed_counter += 1

    def len(self):
        return self.processed_counter

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


dataset = PLPairDataset(root='./refined-set')

# training set과 test set으로 나누기
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
