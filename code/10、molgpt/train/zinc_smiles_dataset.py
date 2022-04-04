import moses
import numpy as np
import openbabel.pybel as pb
import rdkit
import torch



from molegent.atom_alphabet import Atoms
from molegent.molutils import construct_adjacancy_matrix, get_list_atom_types

class ZincDataset:
    def __init__(self, split: str, max_num_atoms: int, max_size: int, shuffle_strategy="no"):

        self.shuffle_strategy = shuffle_strategy

        self._dataset = moses.get_dataset(split)
        self._max_num_atoms = max_num_atoms

        self._max_size = max_size if max_size is not None else float("inf")

    def __len__(self):
        return min(self._max_size, len(self._dataset))

    def __getitem__(self, index):
        smiles = self._dataset[index]

        mol = pb.readstring("smi", smiles)
        mol.removeh()

        atoms, atom_mapping = get_list_atom_types(mol, shuffle=self.shuffle_strategy)
        amat = construct_adjacancy_matrix(mol, atom_mapping=atom_mapping)

        src_atom_seq, src_conn_seq, tgt_atom_seq, tgt_conn_seq = self._prepare_src_tgt_sequence(atoms, amat)

        return_values = {
            "src_atoms": src_atom_seq,
            "src_connections": src_conn_seq,
            "tgt_atoms": tgt_atom_seq,
            "tgt_connections": tgt_conn_seq,
        }

        return_values["smiles"] = smiles

        return return_values

    def _prepare_src_tgt_sequence(self, atoms, adjacency_matrix):
        atoms = torch.tensor(atoms)
        src_atom_seq = torch.cat((torch.tensor([Atoms.BOM.value]), atoms))
        tgt_atom_seq = torch.cat((atoms, torch.tensor([Atoms.EOM.value])))

        amat = np.triu(adjacency_matrix)  #upper triangle of an array
        amat = np.pad(amat, ((0, self._max_num_atoms - amat.shape[0]), (0, 0))).T
        amat = torch.tensor(amat, dtype=torch.float32)

        src_conn_seq = torch.cat((torch.zeros((1, self._max_num_atoms)), amat))
        tgt_conn_seq = torch.cat((amat, torch.zeros((1, self._max_num_atoms))))

        return src_atom_seq, src_conn_seq, tgt_atom_seq, tgt_conn_seq

    def _get_atoms(self, mol):
        atoms = mol.GetAtoms()
        return np.array([Atoms[a.GetSymbol()].value for a in atoms])


def get_zinc_train_and_test_set(
    max_num_atoms, shuffle_strategy, max_size_train=None, max_size_test=None
):

    train_dataset = ZincDataset(
        split="train",
        max_size=max_size_train,
        max_num_atoms=max_num_atoms,
        shuffle_strategy=shuffle_strategy,
    )

    test_dataset = ZincDataset(
        split="test",
        max_size=max_size_test,
        max_num_atoms=max_num_atoms,
        shuffle_strategy=shuffle_strategy,
    )

    return train_dataset, test_dataset
