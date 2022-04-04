import moses
import pandas as pd
import argparse
from rdkit.Chem import Draw
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required = True, help="name of the generated dataset")
    args = parser.parse_args()

    data = pd.read_csv(args.path)

    # test = moses.get_all_metrics(list(data['smiles'].values), device = 'cuda')
    canon_smiles = [canonic_smiles(s) for s in data['smiles']]
    unique_smiles = list(set(canon_smiles))
    mols = []
    for smile in unique_smiles:
        mol = Chem.MolFromSmiles(smile)
        mols.append(mol)
mols_draw = mols[1000:1200]
img = Draw.MolsToGridImage(mols_draw, molsPerRow=5, subImgSize=(300, 300))
img.save("gen_aug_molecular.png")


mols_train = []
data = pd.read_csv('../datasets/' + 'fentn.csv')
data = data.dropna(axis=0).reset_index(drop=True)
train_data = data[data['SPLIT'] == 'train'].reset_index(drop=True)   # 'split' instead of 'source' in moses
canon_smiles_train = [canonic_smiles(s) for s in train_data['SMILES']]
unique_smiles_train = list(set(canon_smiles_train))
for smile in unique_smiles_train:
    mol = Chem.MolFromSmiles(smile)
    mols_train.append(mol)
# img = Draw.MolsToGridImage(mols_train, molsPerRow=5, subImgSize=(300, 300))
# img.save("train_molecular.png")

duplicates = [1 for mol in unique_smiles if mol in unique_smiles_train]  # [1]*45
novel = len(unique_smiles) - sum(duplicates)  # 788-45=743
print(len(novel))


scores=[]
for a in mols_train:
    a=Chem.RDKFingerprint(a)
    for b in mols:
        b=Chem.RDKFingerprint(b)
        score = DataStructs.FingerprintSimilarity(a, b)
        scores.append(score)
print(scores)
