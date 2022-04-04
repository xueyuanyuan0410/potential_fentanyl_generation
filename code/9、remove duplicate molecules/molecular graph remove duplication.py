from rdkit import Chem
from rdkit.Chem.AllChem import CalcNumAtomStereoCenters
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from tqdm import tqdm
def read_smiles(smiles_file):
    """
    Read a list of SMILES from a line-delimited file.
    """
    smiles = []
    with open(smiles_file, 'r') as f:
        smiles.extend([line.strip() for line in f.readlines() \
                       if line.strip()])
    return smiles
def clean_mol(smiles, stereochem=False, selfies=False, deepsmiles=False):
    """
    Construct a molecule from a SMILES string, removing stereochemistry and
    explicit hydrogens, and setting aromaticity.
    """
    if selfies:
        selfies = smiles
        smiles = decoder(selfies)
    elif deepsmiles:
        deepsmiles = smiles
        try:
            smiles = converter.decode(deepsmiles)
        except:
            raise ValueError("invalid DeepSMILES: " + str(deepsmiles))
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid SMILES: " + str(smiles))
    if not stereochem:
        Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol
def clean_mols(all_smiles, stereochem=False, selfies=False, deepsmiles=False):
    """
    Construct a list of molecules from a list of SMILES strings, replacing
    invalid molecules with None in the list.
    """
    mols = []
    for smiles in tqdm(all_smiles):
        try:
            mol = clean_mol(smiles, stereochem, selfies, deepsmiles)
            mols.append(mol)
        except ValueError:
            mols.append(None)
    return mols
sampled_file_1='C:/Users/A/Desktop/generate_moses.txt'
gen_smiles_1 = read_smiles(sampled_file_1)
gen_mols_1 = [mol for mol in clean_mols(gen_smiles_1) if mol]
gen_canonical_1 = [Chem.MolToSmiles(mol) for mol in gen_mols_1]
a=set(gen_canonical_1)
print(len(gen_smiles_1))
print(len(gen_mols_1))
print(len(gen_canonical_1))
print(len(a))
sampled_file_2='H:/研究生期间的科研工作/Exploring the fentanyl chemical space by using deep learning models/与152个已验证芬太尼类物质比对结果/fentanyl.txt'
gen_smiles_2 = read_smiles(sampled_file_2)
gen_mols_2 = [mol for mol in clean_mols(gen_smiles_2) if mol]
gen_canonical_2 = [Chem.MolToSmiles(mol) for mol in gen_mols_2]
b=set(gen_canonical_2)
print(len(gen_smiles_2))
print(len(gen_mols_2))
print(len(gen_canonical_2))
print(len(b))
# with open('C:/Users/A/Desktop/fentanyl analogues generation/code/9、remove duplicate molecules/remove duplicate fentanyl.txt','w')as f:
#     f.write(str(a))
c=a & b
print(c)
print(len(c))
