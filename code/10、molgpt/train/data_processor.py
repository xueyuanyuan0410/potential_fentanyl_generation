from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from itertools import chain
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
# from rdkit.Chem.Draw import IPythonConsole
import re
import pandas as pd
from rdkit.Chem import PandasTools
import functions
from tqdm import tqdm
from functions import clean_mols, read_smiles, \
    continuous_KL, discrete_KL, \
    continuous_JSD, discrete_JSD, \
    continuous_EMD, discrete_EMD, \
    internal_diversity, external_diversity, \
    get_ecfp6_fingerprints

original_file='../data/fentnyl.csv'
org_smiles = read_smiles(original_file)

# #是否需要设置这两个属性:
org_mols = [mol for mol in clean_mols(org_smiles, selfies=False,
                                      deepsmiles=False) if mol]
# for mol in org_mols:
#     Draw.MolToImage(mol, size=(150, 150), kekulize=True)
#     Draw.MolToFile(mol, '../data/output.png', size=(150, 150))
#     test=1

org_canonical = [Chem.MolToSmiles(mol) for mol in org_mols]
# print(org_canonical)
# org_elements=[]
# for mol in org_mols:
#     for atom in mol.GetAtoms():
#         org_elements.append(atom.GetSymbol())
#         test=1

org_elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for \
                     mol in org_mols]
# print('org_elements:')
# print(org_elements)
org_counts = np.unique(list(chain(*org_elements)), return_counts=True)
print('org_counts:')
print(org_counts)
# molecular weights分子量
org_mwt = [Descriptors.MolWt(mol) for mol in org_mols]
# print('org_mwt:')
# print(org_mwt)
# #logP
org_logp = [Descriptors.MolLogP(mol) for mol in org_mols]
# print('org_logp:')
# print(org_logp)
# ## Bertz TC：topological complexity
# org_tcs = [BertzCT(mol) for mol in tqdm(org_mols)]
# # print('org_tcs:')
# # print(org_tcs)
# ## TPSA
org_tpsa = [Descriptors.TPSA(mol) for mol in org_mols]
# # print('org_tpsa:')
# # print(org_tpsa)
# QED
org_qed = []
for mol in org_mols:
    try:
        org_qed.append(Descriptors.qed(mol))
    except OverflowError:
        pass
# print("org_qed")
# print(org_qed)
org_murcko = []
for mol in org_mols:
    try:
        org_murcko.append(MurckoScaffoldSmiles(mol=mol))
    except ValueError:
        pass
# print('org_murcko:')
# print(org_murcko)

writer = Chem.SDWriter('../data/fentn.sdf')
writer.SetProps(['qed', 'logP', 'molwt', 'TPSA', 'scaffold_smiles', 'SPLIT'])
for i, mol in enumerate(org_mols):
    # org_canonical = Chem.MolToSmiles(mol)
    org_mwt  = Descriptors.MolWt(mol)
    org_logp = Descriptors.MolLogP(mol)
    org_tpsa = Descriptors.TPSA(mol)
    org_qed  = Descriptors.qed(mol)
    # org_murcko = MurckoScaffoldSmiles(mol=mol)

    # mol.setProp('SMILES', '%s' %(org_canonical))
    mol.SetProp('qed', '%f' %(org_qed))
    mol.SetProp('logP', '%f' % (org_logp))
    mol.SetProp('molwt', '%f' %(org_mwt))
    mol.SetProp('TPSA', '%f' %(org_tpsa))
    # mol.setProp('scaffold_smiles', '%s' %(org_murcko))
    writer.write(mol)
writer.close()
print('number of mols:', writer.NumMols())
print('mol properties:', [i for i in mol.GetPropNames()])

from rdkit.Chem.ChemUtils.SDFToCSV import Convert
suppl = Chem.SDMolSupplier(r'../data/fentn.sdf')
with open(r'../data/fentnprop.csv', 'w') as csvfile:
    Convert(suppl, csvfile)

data1 = pd.read_csv('../data/fentnprop.csv')
name=['scaffold_smiles']
name1=['SPLIT']
slpit=['train', 'train','train','train','train','train','train','train','train','train','train','train','train','train','train','train','train','test_scaffolds',
       'train', 'train','train','train','train','train','train','train','train','train','train','train','train','train','train','train','train','test_scaffolds',
       'train', 'train','train','train','train','train','train','train','train','train','train','train','train','train','train','train','train','test_scaffolds',
       'test_scaffolds', 'train', 'train','train','train','test_scaffolds','test_scaffolds','test_scaffolds', 'train', 'train','train','train',
       'test_scaffolds','test_scaffolds','test_scaffolds','test_scaffolds','test_scaffolds']
print(len(slpit))
print(len(org_murcko))
data2=pd.DataFrame(columns=name, data=org_murcko)
data3=pd.DataFrame(columns=name1, data=slpit)
dataframe=data1.join(data2)
# dataframe1=dataframe.join(dataframe)
dataframe.to_csv('../data/fentnall.csv', mode='w', index=False)

data4 = pd.read_csv('../data/fentnall.csv')
dataframe1=data4.join(data3)
dataframe1.to_csv('../data/fentnright.csv', mode='w', index=False)


# data['scaffold_smiles']=org_murcko
# data.to_csv('../data/fentnprop.csv', mode='a', index=False)

# create results container
# res = pd.DataFrame()
# res = res.append(pd.DataFrame({
#                 'input_file': original_file,
#                 'outcome': ['qed',
#                             'logP',
#                             'molwt',
#                             'TPSA',
#                             'scaffold_smiles'],
#                 'value': [org_qed, org_mwt, org_logp, org_tpsa, org_murcko] }))
#
# # # write output
# with open('../data/fentnprop.csv', 'w')as f:
#     f.write(str(res))
# # res.to_csv('../data/fentn.csv')


# m=Chem.MolFromSmiles('CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1')
#
# # pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# # regex = re.compile(pattern)
# # smiles = m + str('<')*(54 - len(regex.findall(m.strip())))
# # print(smiles)
# smi_scaffolds = [  MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) for mol in mol_list]
# mol_scaffolds = [Chem.MolFromSmiles(smi_scaffold) for smi_scaffold in smi_scaffolds]



# tpsa_m = Descriptors.TPSA(m)

# print('the TPSA for m is', tpsa_m)
