# -*- coding: utf-8 -*-
"""
Protein PDB parser: extracts atom coordinates, vdW radii, and residue CA info.
Self-contained - no imports from other projects.
"""

import logging
from typing import Tuple, Optional, List

import numpy as np
from Bio.PDB import PDBParser, is_aa

from .utils import VDW_SIGMA, RES2IDX

logger = logging.getLogger(__name__)


def parse_protein_pdb(
    pdb_path: str,
    chain_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a protein PDB file and extract per-atom and per-residue information.

    Parameters
    ----------
    pdb_path : str
        Path to the .pdb file.
    chain_ids : list of str, optional
        If given, only atoms/residues on these chains are returned.
        Each entry is a single character chain ID (e.g. ['A', 'B']).

    Returns
    -------
    atom_pos   : (Na, 3) float32  - Cartesian coordinates of heavy atoms
    atom_sigma : (Na,)   float32  - vdW radius (sigma) per atom
    ca_pos     : (Nr, 3) float32  - Cα positions
    ca_type    : (Nr,)   int16    - residue type index in [0,19], -1 = unknown
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    chain_set = set(chain_ids) if chain_ids is not None else None

    atom_pos: list = []
    atom_sigma: list = []
    ca_pos: list = []
    ca_type: list = []

    for model in structure:
        for chain in model:
            cid = chain.id.strip()
            if chain_set is not None and cid not in chain_set:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                resname = res.get_resname().strip()
                ridx = RES2IDX.get(resname, -1)

                # Cα
                if "CA" in res:
                    ca_pos.append(res["CA"].get_coord().astype(np.float32))
                    ca_type.append(ridx)

                # All heavy atoms with known vdW radius
                for atom in res.get_atoms():
                    elem = atom.element.strip().upper()
                    if not elem or elem not in VDW_SIGMA:
                        continue
                    atom_pos.append(atom.get_coord().astype(np.float32))
                    atom_sigma.append(VDW_SIGMA[elem])

    if len(atom_pos) == 0:
        raise ValueError(f"No valid heavy atoms found in {pdb_path}")

    atom_pos_arr   = np.asarray(atom_pos,   dtype=np.float32)
    atom_sigma_arr = np.asarray(atom_sigma, dtype=np.float32)

    if len(ca_pos) == 0:
        logger.warning("No Cα atoms found in %s; using atom positions as fallback.", pdb_path)
        ca_pos_arr  = atom_pos_arr.copy()
        ca_type_arr = np.full(len(ca_pos_arr), -1, dtype=np.int16)
    else:
        ca_pos_arr  = np.asarray(ca_pos,  dtype=np.float32)
        ca_type_arr = np.asarray(ca_type, dtype=np.int16)

    return atom_pos_arr, atom_sigma_arr, ca_pos_arr, ca_type_arr
