from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
smis_1=[]
for i in open ("after rempve duplication fentnyl.txt"):
    smis_1.append(i)
mols_1= []
for smi_1 in smis_1:
    mol_1 = Chem.MolFromSmiles(smi_1)
    mols_1.append(mol_1)
# img = Draw.MolsToGridImage(mols_1, molsPerRow=5, subImgSize=(300, 300))
# img.save("yuan_molecular.png")

smis_2 = []
for j in open("valid generation for picture.txt"):
    smis_2.append(j)
mols_2 = []
for smi_2 in smis_2:
    mol_2 = Chem.MolFromSmiles(smi_2)
    mols_2.append(mol_2)


scores=[]
for a in mols_1:
    a=Chem.RDKFingerprint(a)
    for b in mols_2:
        b=Chem.RDKFingerprint(b)
        score = DataStructs.FingerprintSimilarity(a, b)
        scores.append(score)


print(scores)

#
# smis=[
#     'CC(C)C(=O)N(C1=CC=CC=C1)C2(CCN(CC2)CCC3=CC=CC=C3)C(=O)OC',
# 'CC(C)C(=O)N(C1=CC=CC=C1)C2(CCN(CC2)CCC3=CC=CC=C3)C(=O)OC',
# ]
# mols =[]
# for smi in smis:
#     m = Chem.MolFromSmiles(smi)
#     mols.append(m)
#
# fps = [Chem.RDKFingerprint(x) for x in mols]
# sm01=DataStructs.FingerprintSimilarity(fps[0],fps[1])

#sm02=DataStructs.FingerprintSimilarity(fps[0],fps[2])

#sm12=DataStructs.FingerprintSimilarity(fps[1],fps[2])
# print("similarity between mol 1 and mol2: %.2f"%sm01)
#print("similarity between mol 1 and mol3: %.2f"%sm02)
#print("similarity between mol 2 and mol3: %.2f"%sm12)