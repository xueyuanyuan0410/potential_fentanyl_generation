"""
Calculate a set of outcomes summarizing the quality of a set of generated
molecules.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import scipy.stats
import sys
from fcd_torch import FCD
from itertools import chain
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem.AllChem import CalcNumAtomStereoCenters
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

# suppress Chem.MolFromSmiles error output
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
import npscorer

import functions
from functions import clean_mols, read_smiles, \
    continuous_KL, discrete_KL, \
    continuous_JSD, discrete_JSD, \
    continuous_EMD, discrete_EMD, \
    internal_diversity, external_diversity, \
    get_ecfp6_fingerprints


# read the training set SMILES, and convert to moelcules
original_file='C:/Users/A/Desktop/fentanyl analogues generation/code/7、Molecular properties and distribution/molecular property calculation/fentanyl.txt'
org_smiles = read_smiles(original_file)
org_mols = [mol for mol in clean_mols(org_smiles) if mol]
org_canonical = [Chem.MolToSmiles(mol) for mol in org_mols]

# define helper function to get # of rotatable bonds
def pct_rotatable_bonds(mol):
    n_bonds = mol.GetNumBonds()
    if n_bonds > 0:
        rot_bonds = Lipinski.NumRotatableBonds(mol) / n_bonds
    else:
        rot_bonds = 0
    return rot_bonds

# define helper function to get % of stereocenters
def pct_stereocentres(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms > 0:
        Chem.AssignStereochemistry(mol)
        pct_stereo = CalcNumAtomStereoCenters(mol) / n_atoms
    else:
        pct_stereo = 0
    return pct_stereo

# calculate training set descriptors

org_elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for \
                     mol in org_mols]
# print('org_elements:')
# print(org_elements)
org_counts = np.unique(list(chain(*org_elements)), return_counts=True)
# print('org_counts:')
# print(org_counts)
## molecular weights
org_mws = [Descriptors.MolWt(mol) for mol in org_mols]
# print('org_mws:')
# print(org_mws)
#logP
org_logp = [Descriptors.MolLogP(mol) for mol in tqdm(org_mols)]
# print('org_logp:')
# print(org_logp)
## Bertz TC：topological complexity
org_tcs = [BertzCT(mol) for mol in tqdm(org_mols)]
# print('org_tcs:')
# print(org_tcs)
## TPSA
org_tpsa = [Descriptors.TPSA(mol) for mol in org_mols]
# print('org_tpsa:')
# print(org_tpsa)
# QED
org_qed = []
for mol in org_mols:
    try:
        org_qed.append(Descriptors.qed(mol))
    except OverflowError:
        pass
# print('org_qed:')
# print(org_qed)
# number of rings
org_rings1 = [Lipinski.RingCount(mol) for mol in tqdm(org_mols)]#环
# print('org_rings1:')
# print(org_rings1)
org_rings2 = [Lipinski.NumAliphaticRings(mol) for mol in tqdm(org_mols)]#脂肪族环
# print('org_rings2:')
# print(org_rings1)
org_rings3 = [Lipinski.NumAromaticRings(mol) for mol in tqdm(org_mols)]#芳香环
# print('org_rings3:')
# print(org_rings3)
# SA score
org_SA = []
for mol in tqdm(org_mols):
    try:
        org_SA.append(sascorer.calculateScore(mol))
    except (OverflowError, ZeroDivisionError):
        pass
# print('org_SA :')
# print(org_SA )
## NP-likeness
fscore = npscorer.readNPModel()
org_NP = [npscorer.scoreMol(mol, fscore) for mol in tqdm(org_mols)]
# print('org_NP:')
# print(org_NP)
# % sp3 carbons
org_sp3 = [Lipinski.FractionCSP3(mol) for mol in org_mols]
# print('org_sp3:')
# print(org_sp3)
## % rotatable bonds
org_rot = [pct_rotatable_bonds(mol) for mol in org_mols]
# print('org_rot:')
# print(org_rot)
## % of stereocentres
org_stereo = [pct_stereocentres(mol) for mol in org_mols]
# print('org_stereo:')
# print(org_stereo)
#Murcko scaffolds
org_murcko = []
for mol in org_mols:
    try:
        org_murcko.append(MurckoScaffoldSmiles(mol=mol))
    except ValueError:
        pass
# print('org_murcko:')
# print(org_murcko)
org_murcko = [MurckoScaffoldSmiles(mol=mol) for mol in org_mols]
org_murcko_counts = np.unique(org_murcko, return_counts=True)
# print('org_murcko_counts:')
# print(org_murcko_counts)
## hydrogen donors/acceptors
org_donors = [Lipinski.NumHDonors(mol) for mol in org_mols]
# print('org_donors:')
# print(org_donors)
org_acceptors = [Lipinski.NumHAcceptors(mol) for mol in org_mols]
# print('org_acceptors:')
# print(org_acceptors)

sampled_file='C:/Users/A/Desktop/fentanyl analogues generation/code/7、Molecular properties and distribution/molecular property calculation/546 valid generated molecules.txt'
gen_smiles = read_smiles(sampled_file)
gen_mols = [mol for mol in clean_mols(gen_smiles) if mol]
gen_canonical = [Chem.MolToSmiles(mol) for mol in gen_mols]
print('gen_smiles_length:')
print(len(gen_smiles))
print('gen_canonical_length:')
print(len(gen_canonical))
# create results container
res = pd.DataFrame()
#calculate descriptors
# outcome 1: % valid
pct_valid = len(gen_mols) / len(gen_smiles)
res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': '% valid',
                'value': [pct_valid] }))

# outcome 2: % novel
#convert back to canonical SMILES for text-based comparison
pct_novel = len([sm for sm in gen_canonical if not sm in \
                         org_canonical]) / len(gen_canonical)
res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': '% novel',
                'value': [pct_novel] }))

# outcome 3: % unique
pct_unique = len(set(gen_canonical)) / len(gen_canonical)
res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': '% unique',
                'value': [pct_unique] }))

## outcome 4: K-L divergence of heteroatom distributions杂原子分布
gen_elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for \
                             mol in gen_mols]
gen_counts = np.unique(list(chain(*gen_elements)),
                                   return_counts=True)
# get all unique keys
keys = np.union1d(org_counts[0], gen_counts[0])
n1, n2 = sum(org_counts[1]), sum(gen_counts[1])
d1 = dict(zip(org_counts[0], org_counts[1]))
d2 = dict(zip(gen_counts[0], gen_counts[1]))
p1 = [d1[key] / n1 if key in d1.keys() else 0 for key in keys]
p2 = [d2[key] / n2 if key in d2.keys() else 0 for key in keys]
# print('p1:')
# print(p1)
# print('p2:')
# print(p2)
kl_atoms = scipy.stats.entropy([p + 1e-10 for p in p2],
                                           [p + 1e-10 for p in p1])
jsd_atoms = jensenshannon(p2, p1)
emd_atoms = wasserstein_distance(p2, p1)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, atoms',
                                'Jensen-Shannon distance, atoms',
                                'Wasserstein distance, atoms'],
                    'value': [kl_atoms, jsd_atoms, emd_atoms] }))

# outcome 5: K-L divergence of molecular weight
gen_mws = [Descriptors.MolWt(mol) for mol in gen_mols]
# print('gen_mws')
# print(gen_mws)
kl_mws = continuous_KL(gen_mws, org_mws)
jsd_mws = continuous_JSD(gen_mws, org_mws)
emd_mws = continuous_EMD(gen_mws, org_mws)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, MWs',
                                'Jensen-Shannon distance, MWs',
                                'Wasserstein distance, MWs'],
                    'value': [kl_mws, jsd_mws, emd_mws] }))

# outcome 6: K-L divergence of LogP
gen_logp = [Descriptors.MolLogP(mol) for mol in gen_mols]
# print(' gen_logp:')
# print( gen_logp)
kl_logp = continuous_KL(gen_logp, org_logp)
jsd_logp = continuous_JSD(gen_logp, org_logp)
emd_logp = continuous_EMD(gen_logp, org_logp)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, logP',
                                'Jensen-Shannon distance, logP',
                                'Wasserstein distance, logP'],
                    'value': [kl_logp, jsd_logp, emd_logp] }))

# outcome 7: K-L divergence of Bertz topological complexity
gen_tcs = [BertzCT(mol) for mol in gen_mols]
# print('gen_tcs')
# print(gen_tcs)
kl_tc = continuous_KL(gen_tcs, org_tcs)
jsd_tc = continuous_JSD(gen_tcs, org_tcs)
emd_tc = continuous_EMD(gen_tcs, org_tcs)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, Bertz TC',
                                'Jensen-Shannon distance, Bertz TC',
                                'Wasserstein distance, Bertz TC'],
                    'value': [kl_tc, jsd_tc, emd_tc] }))

# outcome 8: K-L divergence of QED
gen_qed = []
for mol in gen_mols:
    try:
        gen_qed.append(Descriptors.qed(mol))
    except OverflowError:
        pass
# print("gen_qed")
# print(gen_qed)

kl_qed = continuous_KL(gen_qed, org_qed)
jsd_qed = continuous_JSD(gen_qed, org_qed)
emd_qed = continuous_EMD(gen_qed, org_qed)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, QED',
                                'Jensen-Shannon distance, QED',
                                'Wasserstein distance, QED'],
                    'value': [kl_qed, jsd_qed, emd_qed] }))

# outcome 9: K-L divergence of TPSA
gen_tpsa = [Descriptors.TPSA(mol) for mol in gen_mols]
# print('gen_tpsa:')
# print(gen_tpsa)
kl_tpsa = continuous_KL(gen_tpsa, org_tpsa)
jsd_tpsa = continuous_JSD(gen_tpsa, org_tpsa)
emd_tpsa = continuous_EMD(gen_tpsa, org_tpsa)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, TPSA',
                                'Jensen-Shannon distance, TPSA',
                                'Wasserstein distance, TPSA'],
                    'value': [kl_tpsa, jsd_tpsa, emd_tpsa] }))

# outcome 10: internal diversity
gen_fps = get_ecfp6_fingerprints(gen_mols)
internal_div = internal_diversity(gen_fps)
# print("internal_div")
# print(internal_div)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': 'Internal diversity',
                    'value': [internal_div] }))

# outcome 11: median Tc to original set
org_fps = get_ecfp6_fingerprints(org_mols)
external_div = external_diversity(gen_fps, org_fps)
# print('external_div')
# print(external_div)
res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': 'External diversity',
                'value': [external_div] }))

# outcome 12: K-L divergence of number of rings
gen_rings1 = [Lipinski.RingCount(mol) for mol in gen_mols]
gen_rings2 = [Lipinski.NumAliphaticRings(mol) for mol in gen_mols]
gen_rings3 = [Lipinski.NumAromaticRings(mol) for mol in gen_mols]

kl_rings1 = discrete_KL(gen_rings1, org_rings1)
kl_rings2 = discrete_KL(gen_rings2, org_rings2)
kl_rings3 = discrete_KL(gen_rings3, org_rings3)
# print(kl_rings3)
jsd_rings1 = discrete_JSD(gen_rings1, org_rings1)
jsd_rings2 = discrete_JSD(gen_rings2, org_rings2)
jsd_rings3 = discrete_JSD(gen_rings3, org_rings3)
# print(jsd_rings3)
emd_rings1 = discrete_EMD(gen_rings1, org_rings1)
emd_rings2 = discrete_EMD(gen_rings2, org_rings2)
emd_rings3 = discrete_EMD(gen_rings3, org_rings3)
# print(emd_rings3)
res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': ['KL divergence, # of rings',
                            'KL divergence, # of aliphatic rings',
                            'KL divergence, # of aromatic rings',
                            'Jensen-Shannon distance, # of rings',
                            'Jensen-Shannon distance, # of aliphatic rings',
                            'Jensen-Shannon distance, # of aromatic rings',
                            'Wasserstein distance, # of rings',
                            'Wasserstein distance, # of aliphatic rings',
                            'Wasserstein distance, # of aromatic rings'],
                'value': [kl_rings1, kl_rings2, kl_rings3,
                          jsd_rings1, jsd_rings2, jsd_rings3,
                          emd_rings1, emd_rings2, emd_rings3] }))

# outcome 13: K-L divergence of SA score
gen_SA = []
for mol in gen_mols:
    try:
        gen_SA.append(sascorer.calculateScore(mol))
    except (OverflowError, ZeroDivisionError):
        pass
# print('gen_SA')
# print(gen_SA)
kl_SA = continuous_KL(gen_SA, org_SA)
jsd_SA = continuous_JSD(gen_SA, org_SA)
emd_SA = continuous_EMD(gen_SA, org_SA)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, SA score',
                                'Jensen-Shannon distance, SA score',
                                'Wasserstein distance, SA score'],
                    'value': [kl_SA, jsd_SA, emd_SA] }))

# outcome 14: K-L divergence of NP-likeness
gen_NP = []
for mol in gen_mols:
    try:
        gen_NP.append(npscorer.scoreMol(mol, fscore))
    except (OverflowError, ZeroDivisionError):
        pass
# print('gen_NP')
# print(gen_NP)
kl_NP = continuous_KL(gen_NP, org_NP)
jsd_NP = continuous_JSD(gen_NP, org_NP)
emd_NP = continuous_EMD(gen_NP, org_NP)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, NP score',
                                'Jensen-Shannon distance, NP score',
                                'Wasserstein distance, NP score'],
                    'value': [kl_NP, jsd_NP, emd_NP] }))

# outcome 15: K-L divergence of % sp3 carbons
gen_sp3 = [Lipinski.FractionCSP3(mol) for mol in gen_mols]
# print('gen_sp3')
# print(gen_sp3)
kl_sp3 = continuous_KL(gen_sp3, org_sp3)
jsd_sp3 = continuous_JSD(gen_sp3, org_sp3)
emd_sp3 = continuous_EMD(gen_sp3, org_sp3)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, % sp3 carbons',
                                'Jensen-Shannon distance, % sp3 carbons',
                                'Wasserstein distance, % sp3 carbons'],
                    'value': [kl_sp3, jsd_sp3, emd_sp3] }))

# outcome 16: K-L divergence of % rotatable bonds
gen_rot = [pct_rotatable_bonds(mol) for mol in gen_mols]
# print('gen_rot')
# print(gen_rot)
kl_rot = continuous_KL(gen_rot, org_rot)
jsd_rot = continuous_JSD(gen_rot, org_rot)
emd_rot = continuous_EMD(gen_rot, org_rot)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, % rotatable bonds',
                                'Jensen-Shannon distance, % rotatable bonds',
                                'Wasserstein distance, % rotatable bonds'],
                    'value': [kl_rot, jsd_rot, emd_rot] }))

# outcome 17: K-L divergence of % stereocenters
gen_stereo = [pct_stereocentres(mol) for mol in gen_mols]
# print('gen_stereo')
# print(gen_stereo)
kl_stereo = continuous_KL(gen_stereo, org_stereo)
jsd_stereo = continuous_JSD(gen_stereo, org_stereo)
emd_stereo = continuous_EMD(gen_stereo, org_stereo)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, % stereocenters',
                                'Jensen-Shannon distance, % stereocenters',
                                'Wasserstein distance, % stereocenters'],
                    'value': [kl_stereo, jsd_stereo, emd_stereo] }))

# outcome 18: K-L divergence of Murcko scaffolds
gen_murcko = []
for mol in gen_mols:
    try:
        gen_murcko.append(MurckoScaffoldSmiles(mol=mol))
    except ValueError:
        pass
# gen_murcko = [MurckoScaffoldSmiles(mol=mol) for mol in gen_mols]
gen_murcko_counts = np.unique(gen_murcko, return_counts=True)
# get all unique keys
keys = np.union1d(org_murcko_counts[0], gen_murcko_counts[0])
n1, n2 = sum(org_murcko_counts[1]), sum(gen_murcko_counts[1])
d1 = dict(zip(org_murcko_counts[0], org_murcko_counts[1]))
d2 = dict(zip(gen_murcko_counts[0], gen_murcko_counts[1]))
p1 = [d1[key] / n1 if key in d1.keys() else 0 for key in keys]
p2 = [d2[key] / n2 if key in d2.keys() else 0 for key in keys]
# print('p1')
# print(p1)
# print('p2')
# print(p2)
kl_murcko = scipy.stats.entropy([p + 1e-10 for p in p2],
                                            [p + 1e-10 for p in p1])
jsd_murcko = jensenshannon(p2, p1)
emd_murcko = wasserstein_distance(p2, p1)
res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, Murcko scaffolds',
                                'Jensen-Shannon distance, Murcko scaffolds',
                                'Wasserstein distance, Murcko scaffolds'],
                    'value': [kl_murcko, jsd_murcko, emd_murcko] }))

# outcome 19: K-L divergence of # of hydrogen donors/acceptors
gen_donors = [Lipinski.NumHDonors(mol) for mol in gen_mols]
# print('gen_donors')
# print(gen_donors)
gen_acceptors = [Lipinski.NumHAcceptors(mol) for mol in gen_mols]
# print('gen_acceptors')
# print(gen_acceptors)
kl_donors = discrete_KL(gen_donors, org_donors)
kl_acceptors = discrete_KL(gen_acceptors, org_acceptors)
jsd_donors = discrete_JSD(gen_donors, org_donors)
jsd_acceptors = discrete_JSD(gen_acceptors, org_acceptors)
emd_donors = discrete_EMD(gen_donors, org_donors)
emd_acceptors = discrete_EMD(gen_acceptors, org_acceptors)
res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': ['KL divergence, hydrogen donors',
                            'KL divergence, hydrogen acceptors',
                            'Jensen-Shannon distance, hydrogen donors',
                            'Jensen-Shannon distance, hydrogen acceptors',
                            'Wasserstein distance, hydrogen donors',
                            'Wasserstein distance, hydrogen acceptors'],
                'value': [kl_donors, kl_acceptors,
                          jsd_donors, jsd_acceptors,
                          emd_donors, emd_acceptors] }))

# outcome 20: Frechet ChemNet distance
fcd = FCD(canonize=False)
fcd_calc = fcd(gen_canonical, org_canonical)
# print('fcd_calc')
# print(fcd_calc)

res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': 'Frechet ChemNet distance',
                'value': [fcd_calc] }))
print(res)

#write output
with open('C:/Users/A/Desktop/outcome.txt','w')as f:
    f.write(str(res))
#res.to_csv('C:/Users/A/Desktop/outcome.csv')
