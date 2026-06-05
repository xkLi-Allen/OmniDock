# -*- coding: utf-8 -*-
"""
Ligand parser: reads small-molecule structures from .sdf or .mol2 via RDKit.
Outputs atom positions, types, bond graph, and scalar molecular features.
Self-contained - no imports from other projects.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

from .utils import ELEMENT_LIST, ELEMENT2IDX, VDW_SIGMA

logger = logging.getLogger(__name__)

# RDKit bond type -> integer
BOND_TYPE_MAP: Dict[Any, int] = {}
if RDKIT_OK:
    BOND_TYPE_MAP = {
        Chem.rdchem.BondType.SINGLE:    0,
        Chem.rdchem.BondType.DOUBLE:    1,
        Chem.rdchem.BondType.TRIPLE:    2,
        Chem.rdchem.BondType.AROMATIC:  3,
    }

# Hybridisation -> int
HYBRID_MAP: Dict[Any, int] = {}
if RDKIT_OK:
    HYBRID_MAP = {
        Chem.rdchem.HybridizationType.SP:    0,
        Chem.rdchem.HybridizationType.SP2:   1,
        Chem.rdchem.HybridizationType.SP3:   2,
        Chem.rdchem.HybridizationType.SP3D:  3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }


def _element_index(symbol: str) -> int:
    sym = symbol.strip().upper()
    return ELEMENT2IDX.get(sym, ELEMENT2IDX["OTHER"])


def _element_sigma(symbol: str) -> float:
    sym = symbol.strip().upper()
    # title-case lookup matches VDW_SIGMA keys
    return VDW_SIGMA.get(sym, 1.70)


def _mol_to_dict(mol) -> Dict[str, np.ndarray]:
    """
    Convert an RDKit Mol (with 3D conformer) to the canonical ligand dict.

    Returns
    -------
    dict with keys:
        lig_pos           (Na, 3)  float32  - atom 3D positions
        lig_atom_type     (Na,)    int32    - element index
        lig_atom_element  (Na,)    int32    - same as lig_atom_type (kept for API compat)
        lig_atom_charge   (Na,)    float32  - formal charge
        lig_atom_aromatic (Na,)    int8     - is_aromatic flag
        lig_atom_hybrid   (Na,)    int8     - hybridisation index
        lig_atom_numHs    (Na,)    int8     - num implicit+explicit Hs
        lig_atom_sigma    (Na,)    float32  - vdW radius per atom
        lig_edge_index    (2, Nb)  int32    - bond src/dst (both directions)
        lig_edge_type     (Nb,)    int32    - bond type (0=single,...,3=aromatic)
        lig_center        (3,)     float32  - geometric centre of ligand
    """
    conf = mol.GetConformer()
    positions = conf.GetPositions().astype(np.float32)   # (Na, 3)

    Na = mol.GetNumAtoms()
    atom_type    = np.zeros(Na, dtype=np.int32)
    atom_element = np.zeros(Na, dtype=np.int32)
    atom_charge  = np.zeros(Na, dtype=np.float32)
    atom_aromatic = np.zeros(Na, dtype=np.int8)
    atom_hybrid  = np.zeros(Na, dtype=np.int8)
    atom_numHs   = np.zeros(Na, dtype=np.int8)
    atom_sigma   = np.zeros(Na, dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        eidx = _element_index(sym)
        atom_type[i]    = eidx
        atom_element[i] = eidx
        atom_charge[i]  = float(atom.GetFormalCharge())
        atom_aromatic[i] = int(atom.GetIsAromatic())
        atom_hybrid[i]  = HYBRID_MAP.get(atom.GetHybridization(), -1)
        atom_numHs[i]   = atom.GetTotalNumHs()
        atom_sigma[i]   = _element_sigma(sym)

    # Build bond index (undirected -> store both directions)
    src_list, dst_list, type_list = [], [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        btype = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        src_list += [u, v]
        dst_list += [v, u]
        type_list += [btype, btype]

    if src_list:
        edge_index = np.array([src_list, dst_list], dtype=np.int32)
        edge_type  = np.array(type_list, dtype=np.int32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_type  = np.empty((0,),   dtype=np.int32)

    center = positions.mean(axis=0).astype(np.float32)

    return dict(
        lig_pos           = positions,
        lig_atom_type     = atom_type,
        lig_atom_element  = atom_element,
        lig_atom_charge   = atom_charge,
        lig_atom_aromatic = atom_aromatic,
        lig_atom_hybrid   = atom_hybrid,
        lig_atom_numHs    = atom_numHs,
        lig_atom_sigma    = atom_sigma,
        lig_edge_index    = edge_index,
        lig_edge_type     = edge_type,
        lig_center        = center,
    )


def parse_ligand(
    sdf_path: Optional[str] = None,
    mol2_path: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Parse a small-molecule ligand.  Tries SDF first, then mol2.

    Parameters
    ----------
    sdf_path  : path to .sdf file (or None)
    mol2_path : path to .mol2 file (or None)

    Returns
    -------
    (ligand_dict, source_type)
        ligand_dict  : see _mol_to_dict
        source_type  : 'sdf' | 'mol2'

    Raises
    ------
    ValueError if both formats fail.
    """
    if not RDKIT_OK:
        raise ImportError("RDKit is required for ligand parsing.")

    errors = []

    # --- try SDF ---
    if sdf_path:
        try:
            supp = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
            mol = next((m for m in supp if m is not None), None)
            if mol is not None and mol.GetNumConformers() > 0:
                return _mol_to_dict(mol), "sdf"
            errors.append(f"SDF supplier returned None or no conformer: {sdf_path}")
        except Exception as exc:
            errors.append(f"SDF parse error ({sdf_path}): {exc}")

    # --- try mol2 ---
    if mol2_path:
        try:
            mol = Chem.MolFromMol2File(mol2_path, removeHs=True, sanitize=True)
            if mol is not None and mol.GetNumConformers() > 0:
                return _mol_to_dict(mol), "mol2"
            errors.append(f"mol2 returned None or no conformer: {mol2_path}")
        except Exception as exc:
            errors.append(f"mol2 parse error ({mol2_path}): {exc}")

    raise ValueError("Ligand parse failed.\n" + "\n".join(errors))
