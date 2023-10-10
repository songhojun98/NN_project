# PDBbind v.2020

![filename_all](https://github.com/songhojun98/NN_project/assets/141312067/449bb871-c059-4c9d-b082-1636ac913ecd)

## Introduction
The aim of the PDBbind database is to provide a comprehensive collection of experimentally measured binding affinity data for all biomolecular complexes deposited in the Protein Data Bank (PDB). It provides an essential linkage between the energetic and structural information of those complexes, which is helpful for various computational and statistical studies on molecular recognition, drug discovery, and many more (see the list of published applications of PDBbind).

The PDBbind database was originally developed by Prof. Shaomeng Wang's group at the University of Michigan in USA, which was first released to the public in May, 2004. This database is now maintained and further developed by Prof. Renxiao Wang's group at College of Pharmacy, Fudan University in China. The PDBbind database is updated on an annual base to keep up with the growth of the Protein Data Bank.

## Installation

### Train/Val dataset
````
wget http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz
wget http://www.pdbbind.org.cn/download/PDBbind_v2020_plain_text_index.tar.gz
````
PDBbind_v2020_refined:
> Data package of the refined set, including index files summarizing the basic information and processed structural files for the protein-ligand complexes included in this data set (proteins saved in PDB format; ligands saved in Mol2 and SDF format). This data package includes 5316 complexes in total.
PDBbind_v2020_plain_text_index: 
> Index files in plain text, which summarize the basic information (e.g. PDB code, resolution, release year, binding data etc.) of the complexes recorded in PDBbind.
### Test dataset
````
wget http://www.pdbbind.org.cn/download/CASF-2016.tar.gz
````
CASF-2016 dataset: 
> The latest available version of the PDBbind core set is included in CASF-2016, which consists of 285 protein-ligand complexes
    
### Notes:

> Protein PDB files are assumed to contain coordinates for all heavy atoms
    
## Presentation Video
> https://drive.google.com/file/d/1fqrfNMa_-avkyksGpn99XE1gID2vTAfm/view?usp=drive_link

## Contect information
> Phone number: 010-7529-0763
> Email: h_j_song@korea.ac.kr
