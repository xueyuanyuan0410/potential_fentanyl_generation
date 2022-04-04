from __future__ import absolute_import, division, print_function
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import  MolFromSmiles, MolToSmiles

def verified_and_below(smile, max_len):
    return len(smile) < max_len and verify_sequence(smile)

def verify_sequence(smile):#
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1

with open('C:/Users/A/Desktop/fentanyl analogues generation/code/4、screening valid molecules/remove_fentnyl new generate fentnyl data.txt')as f:
    samples=f.readlines()
    
verified_samples = [
        sample for sample in samples if verify_sequence(sample)]
        
with open('C:/Users/A/Desktop/fentanyl analogues generation/code/4、screening valid molecules/valid generation.txt','w') as f:
    f.write(str(verified_samples))
    f.write(str(len(verified_samples)))
