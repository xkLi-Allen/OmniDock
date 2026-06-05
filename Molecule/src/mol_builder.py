# -*- coding: utf-8 -*-
"""
Rebuild a valid RDKit molecule from generated atom positions,
element types, and a predicted bond-type matrix.
Outputs SMILES and/or SDF.  Self-contained (only RDKit + numpy).
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ELEMENT_LIST = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H", "C"]
# Note: last entry is fallback for OTHER -> treat as C


def _rdkit_bond_type(bt: int):
    from rdkit.Chem import BondType
    return {
        1: BondType.SINGLE,
        2: BondType.DOUBLE,
        3: BondType.TRIPLE,
        4: BondType.AROMATIC,
    }.get(bt, BondType.SINGLE)


def build_molecule(
    atom_pos:    np.ndarray,
    atom_types:  np.ndarray,
    bond_matrix: np.ndarray,
    sanitize:    bool = True,
    remove_Hs:   bool = False,
):
    """
    Build an RDKit Mol from raw generation output.

    Parameters
    ----------
    atom_pos    : (Na, 3) float32 -- absolute coordinates in Angstroms
    atom_types  : (Na,)   int    -- indices into ELEMENT_LIST
    bond_matrix : (Na, Na) int   -- symmetric; 0=none 1-4=bond type

    Returns
    -------
    rdkit.Chem.Mol or None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
        from rdkit.Geometry import Point3D
    except ImportError:
        logger.error("RDKit not available.")
        return None

    Na = atom_pos.shape[0]
    rw = RWMol()

    for i in range(Na):
        idx = int(atom_types[i])
        sym = ELEMENT_LIST[idx] if idx < len(ELEMENT_LIST) else "C"
        a = Chem.Atom(sym)
        a.SetNoImplicit(False)
        rw.AddAtom(a)

    for i in range(Na):
        for j in range(i + 1, Na):
            bt = int(bond_matrix[i, j])
            if bt == 0:
                continue
            try:
                rw.AddBond(i, j, _rdkit_bond_type(bt))
            except Exception:
                pass

    conf = Chem.Conformer(Na)
    for i in range(Na):
        conf.SetAtomPosition(
            i, Point3D(float(atom_pos[i, 0]),
                       float(atom_pos[i, 1]),
                       float(atom_pos[i, 2])))
    rw.AddConformer(conf, assignId=True)
    mol = rw.GetMol()

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.warning("Sanitization failed: %s", e)

    if remove_Hs:
        try:
            from rdkit.Chem import RemoveHs
            mol = RemoveHs(mol)
        except Exception:
            pass

    return mol


def distance_based_bonds(
    atom_pos: np.ndarray,
    single_cutoff: float = 1.9,
    double_cutoff: float = 1.4,
    triple_cutoff: float = 1.25,
) -> np.ndarray:
    """Heuristic bond matrix from interatomic distances (fallback)."""
    Na = atom_pos.shape[0]
    bm = np.zeros((Na, Na), dtype=np.int32)
    for i in range(Na):
        for j in range(i + 1, Na):
            d = float(np.linalg.norm(atom_pos[i] - atom_pos[j]))
            if d <= triple_cutoff:
                bt = 3
            elif d <= double_cutoff:
                bt = 2
            elif d <= single_cutoff:
                bt = 1
            else:
                bt = 0
            bm[i, j] = bm[j, i] = bt
    return bm


def build_molecule_robust(
    atom_pos:    np.ndarray,
    atom_types:  np.ndarray,
    bond_matrix: np.ndarray,
    fallback_to_distance: bool = True,
):
    """
    Try neural bond_matrix first; if molecule is invalid fall back
    to distance-based heuristic bonds.
    """
    mol = build_molecule(atom_pos, atom_types, bond_matrix, sanitize=True)
    if mol is not None:
        return mol
    if fallback_to_distance:
        logger.info("Neural bonds invalid; falling back to distance-based bonds.")
        bm2 = distance_based_bonds(atom_pos)
        mol = build_molecule(atom_pos, atom_types, bm2, sanitize=True)
    return mol


def mol_to_smiles(mol) -> Optional[str]:
    if mol is None:
        return None
    try:
        from rdkit import Chem
        smi = Chem.MolToSmiles(mol)
        return smi or None
    except Exception as e:
        logger.warning("MolToSmiles failed: %s", e)
        return None


def mol_to_sdf(mol, path: str) -> bool:
    if mol is None:
        return False
    try:
        from rdkit.Chem import SDWriter
        with SDWriter(path) as w:
            w.write(mol)
        return True
    except Exception as e:
        logger.error("SDF write failed: %s", e)
        return False
