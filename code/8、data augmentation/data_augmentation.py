"""
Take an input SMILES file, and augment it by some fixed factor via SMILES
enumeration.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import threading


#from functions import read_smiles, write_smiles
def read_smiles(smiles_file):
    """
    Read a list of SMILES from a line-delimited file.
    """
    smiles = []
    with open(smiles_file, 'r') as f:
        smiles.extend([line.strip() for line in f.readlines() \
                       if line.strip()])
    return smiles

def write_smiles(smiles, smiles_file):
    """
    Write a list of SMILES to a line-delimited file.
    """
    # write sampled SMILES
    with open(smiles_file, 'w') as f:
        for sm in smiles:
            _ = f.write(sm + '\n')


class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer

    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        leftpad: Add spaces to the left of the SMILES
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    """

    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True,
                 isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)




input_file='C:/Users/A/Desktop/fentanyl analogues generation/code/8、data augmentation/fentanyl.txt'
output_file='C:/Users/A/Desktop/fentanyl analogues generation/code/8、data augmentation/data augment fentnyl.txt'
# read SMILES
smiles = read_smiles(input_file)
# convert to numpy array
smiles = np.asarray(smiles)
enum_factor=10
# create enumerator
sme = SmilesEnumerator(canonical=False, enum=True)

# also store and write information about enumerated library size
summary = pd.DataFrame()

# enumerate potential SMILES
enum = []
max_tries = 200 ## randomized SMILES to generate for each input structure
for sm_idx, sm in enumerate(tqdm(smiles)):
    tries = []
    for try_idx in range(max_tries):
        this_try = sme.randomize_smiles(sm)
        tries.append(this_try)
        tries = [rnd for rnd in np.unique(tries)]
        if len(tries) > enum_factor:
            tries = tries[:enum_factor]
            break
    enum.extend(tries)

# write to line-delimited file
write_smiles(enum, output_file)
print("wrote " + str(len(enum)) + " SMILES to output file: " + \
      output_file)
