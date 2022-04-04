from rdkit import Chem
from rdkit.Chem import Draw


smis=[]
for i in open ("C:/Users/A/Desktop/fentanyl analogues generation/code/5、draw molecular picture/decode_data.txt"):
    smis.append(i)


mols = []
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    mols.append(mol)

mols = mols[100:150]

# output picture
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300))
img.save("C:/Users/A/Desktop/101-150.png")



