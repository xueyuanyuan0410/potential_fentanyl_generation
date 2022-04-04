from __future__ import absolute_import, division, print_function
import os
import numpy as np
import csv
from rdkit import rdBase
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import  MolFromSmiles, MolToSmiles
from math import exp, log
# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')
#====== load data



def load_training_set( file):
    """Specifies a training set for the model. It also finishes
    the model set up, as some of the internal parameters require
    knowledge of the vocabulary.

    Arguments
    -----------

        - file. String pointing to the dataset file.

    """

    # Load training set
    train_samples = load_train_data(file)
    # Process and create vocabulary
    char_dict, ord_dict = build_vocab(train_samples)
    NUM_EMB = len(char_dict)
    PAD_CHAR = ord_dict[NUM_EMB - 1]
    PAD_NUM = char_dict[PAD_CHAR]
    DATA_LENGTH = max(map(len, train_samples))
    print('Vocabulary:')
    print(list(char_dict.keys()))
    # If MAX_LENGTH has not been specified by the user, it
    # will be set as 1.5 times the maximum length in the
    # trining set.

    MAX_LENGTH = int(len(max(train_samples, key=len)) * 1.5)

    # Encode samples
    to_use = [sample for sample in train_samples
              if verified_and_below(sample, MAX_LENGTH)]
    print(to_use)
    print(len(to_use))
    positive_samples = [encode(sam,
                                MAX_LENGTH,
                                char_dict) for sam in to_use]
    print(positive_samples)
    with open('C:/Users/A/Desktop/potential_fentanyl_generation/potential_fentanyl_generation/code/1、encoding/data.txt','w') as file_object:
        #The above path is the path of the output file
        for i in positive_samples:
            output=str(i)
            file_object.write(output + '\n')


    POSITIVE_NUM = len(positive_samples)

def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def constant_bump(x, x_low, x_high, decay=0.025):
    if x <= x_low:
        return np.exp(-(x - x_low)**2 / decay)
    elif x >= x_high:
        return np.exp(-(x - x_high)**2 / decay)
    else:
        return 1
    return

def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)

#====== encoding/decoding utility

def canon_smile(smile):
    return MolToSmiles(MolFromSmiles(smile))

def verified_and_below(smile, max_len):
    return len(smile) < max_len and verify_sequence(smile)

def verify_sequence(smile):#
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1

def build_vocab(smiles=None, pad_char='_', start_char='^'):
    # smile syntax
    chars = []
    # atoms (carbon), replace Cl for Q and Br for W
    chars = chars + ['H', 'B', 'c', 'C', 'n', 'N',
                     'o', 'O', 'p', 'P', 's', 'S', 'F', 'Q', 'W', 'I']
    # Atom modifiers: negative charge - has been replaced with ~
    # added explicit hidrogens as Z (H2) and X (H3)
    # negative charge ~ (-), ! (-2),,'&' (-3)
    # positive charge +, u (+2), y (+3)
    chars = chars + ['[', ']', '+', 'u', 'y', '~', '!', '&', 'Z', 'X']
    # bonding
    chars = chars + ['-', '=', '#','.']
    # branches
    chars = chars + ['(', ')']
    # cycles
    chars = chars + ['1', '2', '3', '4', '5', '6', '7','8' ]
    # anit/clockwise
    chars = chars + ['@']
    # directional bonds
    chars = chars + ['/', '\\']

    char_dict = {}
    char_dict[start_char] = 0
    for i, c in enumerate(chars):
        char_dict[c] = i + 1
    # end and start
    char_dict[pad_char] = i + 2

    ord_dict = {v: k for k, v in char_dict.items()}
    print(ord_dict,char_dict)

    return char_dict, ord_dict

def pad(smi, n, pad_char='_'):
    if n < len(smi):
        return smi
    return smi + pad_char * (n - len(smi))

def encode(smi, max_len, char_dict):
    # replace double char atoms symbols
    smi = smi.replace('Cl', 'Q')
    smi = smi.replace('Br', 'W')

    atom_spec = False
    new_chars = [''] * max_len
    i = 0
    for c in smi:
        if c == '[':
            atom_spec = True
            spec = []
        if atom_spec:
            spec.append(c)
        else:
            new_chars[i] = c
            i = i + 1
        # close atom spec
        if c == ']':
            atom_spec = False
            spec = ''.join(spec)
            # negative charges
            spec = spec.replace('-3', '&')
            spec = spec.replace('-2', '!')
            spec = spec.replace('-', '~')
            # positive charges
            spec = spec.replace('+3', 'y')
            spec = spec.replace('+2', 'u')
            # hydrogens
            spec = spec.replace('H2', 'Z')
            spec = spec.replace('H3', 'X')

            new_chars[i:i + len(spec)] = spec
            i = i + len(spec)

    new_smi = ''.join(new_chars)
    return [char_dict[c] for c in pad(new_smi, max_len)]

def load_train_data(filename):
    ext = filename.split(".")[-1]
    if ext == 'csv':
        return read_smiles_csv(filename)
    if ext == 'smi':
        return read_smi(filename)
    else:
        raise ValueError('data is not smi or csv!')
    return

def read_smiles_csv(filename):
    # Assumes smiles is in column 0
    with open(filename) as file:
        reader = csv.reader(file)
        smiles_idx = next(reader).index("smiles")
        data = [row[smiles_idx] for row in reader]
    return data

def save_smi(name, smiles):
    if not os.path.exists('epoch_data'):
        os.makedirs('epoch_data')
    smi_file = os.path.join('epoch_data', "{}.smi".format(name))
    with open(smi_file, 'w') as afile:
        afile.write('\n'.join(smiles))
    return

def read_smi(filename):
    with open(filename) as file:
        smiles = file.readlines()
    smiles = [i.strip() for i in smiles]
    return smiles

load_training_set('C:/Users/A/Desktop/potential_fentanyl_generation/potential_fentanyl_generation/code/1、encoding/data.csv')
#The above path is the path of the input file
