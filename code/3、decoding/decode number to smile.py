from __future__ import absolute_import, division, print_function
import numpy as np
from rdkit import rdBase
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles
from math import exp, log
# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')
#====== load data

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
    print('ord_dict',ord_dict)
    print('char_dict', char_dict)

    return char_dict, ord_dict

def unpad(smi, pad_char='_'):
    return smi.rstrip(pad_char)


def decode(ords, ord_dict):

    smi = unpad(''.join([ord_dict[o] for o in ords]))
    # negative charges
    smi = smi.replace('~', '-')
    smi = smi.replace('!', '-2')
    smi = smi.replace('&', '-3')
    # positive charges
    smi = smi.replace('y', '+3')
    smi = smi.replace('u', '+2')
    # hydrogens
    smi = smi.replace('Z', 'H2')
    smi = smi.replace('X', 'H3')
    # replace proxy atoms for double char atoms symbols
    smi = smi.replace('Q', 'Cl')
    smi = smi.replace('W', 'Br')

    return smi




char_dict, ord_dict = build_vocab()
#The file is the generated sequence of numbers
samples=np.loadtxt('C:/Users/A/Desktop/fentanyl analogues generation/code/3、decoding/final2.txt')
decoded_raw= [decode(sample, ord_dict)
              for sample in samples]
with open('C:/Users/A/Desktop/fentanyl analogues generation/code/3、decoding/decode_data_1.txt','w')as f:
    for line in decoded_raw:
        f.write(line+'\n')
