import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re
import random
from rdkit import Chem
from rdkit.Chem import Draw

def randomSmiles(mol):
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:#有些smiles数据无法转化为分子图结构
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

cnt = 0
def smile_augmentation(smile, augmentation, max_len, sca, prop):
    global cnt
    mol = Chem.MolFromSmiles(smile)
    s = set()
    scas = []
    props = []
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    for i in range(1000):
        mols = []
        smiles = randomSmiles(mol)
        if len(smiles) <= max_len:
            smiles = smiles + str('<') * (max_len - len(regex.findall(smiles.strip())))
            s.add(smiles)

            if len(s) == augmentation:
                # for temp_smile in s:
                #     temp_smile = temp_smile.replace('<', '')
                #     mol = get_mol(temp_smile)
                #     mols.append(mol)
                # cnt=cnt+1
                # if len(mols) > 0:
                #     img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300))
                #     img.save("./aug/aug_molecular_{}.png".format(cnt))
                break

    for i in range(len(s)):
        scas.append(sca)#做了数据增强后的分子仍是同一个分子，所以性质和骨架都是一样的。
        props.append(prop)

    return list(s), scas, props

class SmileDataset(Dataset):
    def __init__(self, args, data, content, block_size, aug_prob = 0.5, augmentation=0, prop = None, scaffold = None, scaffold_maxlen = None):
        chars = sorted(list(set(content)))
        #content是字典里的内容
        data_size, vocab_size = len(data), len(chars)
        #vocab_size是字典的长度
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        #i是索引，ch是索引对应的字符
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        # self.data = data
        # self.prop = prop
        # self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        # self.tfm = SmilesEnumerator()
        self.tfm = SmilesEnumerator(content)
        print('self.tfm')
        print(self.tfm)
        self.aug_prob = aug_prob

        if augmentation > 0:#数据增强
            train_aug, train_scaffold, train_props = self.augment(data, augmentation, block_size, scaffold, prop)
            temp_data = data + train_aug
            temp_scaffold = scaffold + train_scaffold
            temp_props = prop + train_props
            self.data = list(temp_data)
            self.sca = list(temp_scaffold)
            self.prop = list(temp_props)
        else:
            self.data = data
            self.prop = prop
            self.sca  = scaffold

    def augment(self, data, augmentation, max_len, scaffold, prop):
        all_alternative_smi = []
        all_scaffold_smi = []
        all_props = []
        for i, x in enumerate(data):
            x = x.replace('<', '')
            alternative_smi, scas, props = smile_augmentation(x, augmentation, max_len, scaffold[i], prop[i])
            all_scaffold_smi.extend(scas)
            all_alternative_smi.extend(alternative_smi)
            all_props.extend(props)
        return all_alternative_smi, all_scaffold_smi, all_props

    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def  one_hot_encode(self, token_list, n_chars):
        output = np.zeros((len(token_list), n_chars))
        for j, token in enumerate(token_list):
            output[j, token] = 1
        return output

    def __getitem__(self, idx):
        smiles, prop, scaffold = self.data[idx], self.prop[idx], self.sca[idx]    # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = scaffold.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            aug_smiles = smiles
            aug_smiles = aug_smiles.replace('<', '')
            smiles = self.tfm.randomize_smiles(aug_smiles)

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<')*(self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles=regex.findall(smiles)

        scaffold += str('<')*(self.scaf_max_len - len(regex.findall(scaffold)))
        
        if len(regex.findall(scaffold)) > self.scaf_max_len:
            scaffold = scaffold[:self.scaf_max_len]

        scaffold=regex.findall(scaffold)

        dix =  [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold]

        # #one hot encode for smiles
        # dix_one_hot = self.one_hot_encode(dix, self.vocab_size)
        # sca_dix_one_hot = self.one_hot_encode(sca_dix, self.vocab_size)
        # sca_tensor_one_hot = torch.tensor(sca_dix_one_hot, dtype=torch.long)
        # x = torch.tensor(dix_one_hot[:-1, ], dtype=torch.long)
        # y = torch.tensor(dix_one_hot[1:, ], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.float)
        #
        # return x, y, prop, sca_tensor_one_hot

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        prop = torch.tensor([prop], dtype = torch.float)
        return x, y, prop, sca_tensor
