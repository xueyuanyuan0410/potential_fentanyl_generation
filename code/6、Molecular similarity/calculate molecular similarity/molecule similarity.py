from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
smis_1=[]
for i in open ("C:/Users/A/Desktop/fentanyl analogues generation/code/6、Molecular similarity/calculate molecular similarity/after remove duplication fentnyl.txt"):
    smis_1.append(i)
mols_1= []
for smi_1 in smis_1:
    mol_1 = Chem.MolFromSmiles(smi_1)
    mols_1.append(mol_1)


smis_2 = []
for j in open("C:/Users/A/Desktop/fentanyl analogues generation/code/6、Molecular similarity/calculate molecular similarity/valid generation for picture.txt"):
    smis_2.append(j)
mols_2 = []
for smi_2 in smis_2:
    mol_2 = Chem.MolFromSmiles(smi_2)
    mols_2.append(mol_2)


scores=[]
for b in mols_2:
    b=Chem.RDKFingerprint(b)
    score = DataStructs.FingerprintSimilarity(a, b)
    scores.append(score)



print(scores)

with open('molecular similarity.txt','w')as f:
    f.write(str(scores))