# -*- coding: utf-8 -*-
"""
Guided Ligand Generation: base generator + Stage-3 Gradient Inversion
"""

from __future__ import annotations
import argparse, logging, sys, os
from pathlib import Path
import numpy as np
import torch
from torch_scatter import scatter_mean

BASE_GENERATOR_DIR = os.environ.get("BASE_GENERATOR_DIR", str(Path.home() / "molecule_generator"))
PROJECT_DIR        = os.environ.get("PROJECT_DIR", str(Path.home() / "inversiondock_mole"))
sys.path.insert(0, BASE_GENERATOR_DIR)
sys.path.insert(0, PROJECT_DIR)

from openbabel import openbabel
openbabel.obErrorLog.StopLogging()
import utils as generator_utils
from lightning_modules import LigandPocketDDPM
from src.models import DockingModel

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def load_stage3(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = DockingModel(d_model=256, nhead=8, nlayers=6, lig_layers=3, dropout=0.1)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    logger.info("[Stage3] Loaded from %s", ckpt_path)
    return model


def load_npz_pocket(npz_path, seq_len=512, K=32, device=torch.device("cpu")):
    import json
    with np.load(npz_path, allow_pickle=True) as d:
        xs      = d["xs"].astype(np.float32)
        ns      = d["ns"].astype(np.float32)
        centers = d["patch_centers"].astype(np.float32)
        knn     = d["patch_knn_idx"].astype(np.int64)
        order   = d["patch_order"].astype(np.int64)
        lig_center = d["lig_center"].astype(np.float32) if "lig_center" in d else centers.mean(axis=0).astype(np.float32)

    K0 = knn.shape[1]
    if K0 < K:
        knn = np.concatenate([knn, np.tile(knn[:, -1:], (1, K - K0))], axis=1)
    elif K0 > K:
        knn = knn[:, :K]

    Nc = centers.shape[0]
    rng = np.random.default_rng(2024)
    sel = order if Nc <= seq_len else order[int(rng.integers(0, Nc - seq_len + 1)):int(rng.integers(0, Nc - seq_len + 1)) + seq_len]
    sel = order[:seq_len] if Nc > seq_len else order

    pts_idx = knn[sel]
    ctrs    = centers[sel]
    rel_xyz = xs[pts_idx] - ctrs[:, None, :]
    norms   = ns[pts_idx]
    feats   = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)

    rec_feats     = torch.from_numpy(feats).unsqueeze(0).to(device)
    rec_centers   = torch.from_numpy(ctrs).unsqueeze(0).to(device)
    rec_mask      = torch.zeros(1, feats.shape[0], dtype=torch.bool, device=device)
    pocket_center = torch.from_numpy(lig_center).unsqueeze(0).to(device)

    return rec_feats, rec_centers, rec_mask, pocket_center


def compute_guidance_grad(stage3, rec_feats, rec_centers, rec_mask,
                          pocket_center, z_lig, lig_mask, n_samples,
                          guidance_scale, norm_std=10.0):
    """
    Compute Stage-3 affinity gradient w.r.t. ligand coordinates.
    """
    coords_ang = z_lig[:, :3].detach().contiguous() * norm_std
    coords_ang = coords_ang.contiguous().requires_grad_(True)

    sample_sizes = [(lig_mask == i).sum().item() for i in range(n_samples)]
    Na_max = max(sample_sizes) if sample_sizes else 1

    lig_pos_b  = torch.zeros(n_samples, Na_max, 3, dtype=coords_ang.dtype,
                             device=coords_ang.device)
    lig_type_b = torch.zeros(n_samples, Na_max, dtype=torch.long,
                             device=coords_ang.device)
    lig_pad_b  = torch.ones(n_samples, Na_max, dtype=torch.bool,
                            device=coords_ang.device)

    for i in range(n_samples):
        sel = (lig_mask == i)
        n   = sel.sum().item()
        if n == 0:
            continue
        lig_pos_b[i, :n] = coords_ang[sel] - pocket_center[0]
        lig_pad_b[i, :n] = False

    ei_list = [torch.zeros(2, 0, dtype=torch.long, device=coords_ang.device)
               for _ in range(n_samples)]
    et_list = [torch.zeros(0, dtype=torch.long, device=coords_ang.device)
               for _ in range(n_samples)]

    rf = rec_feats.expand(n_samples, -1, -1, -1)
    rc = rec_centers.expand(n_samples, -1, -1)
    rm = rec_mask.expand(n_samples, -1)

    with torch.enable_grad():
        out   = stage3(rf, rc, rm, lig_pos_b, lig_type_b,
                       ei_list, et_list, lig_pad_b)
        score = out["affinity_pred"].sum()
        grad  = torch.autograd.grad(score, coords_ang)[0]

    grad_norm = grad / (norm_std + 1e-8)

    grad_out = torch.zeros_like(z_lig[:, :3])
    for i in range(n_samples):
        sel = (lig_mask == i)
        g   = grad_norm[sel]
        g   = g - g.mean(dim=0, keepdim=True)
        grad_out[sel] = g

    return guidance_scale * grad_out


def decode_snapshot(z_lig, xh_pocket, lig_mask, pocket_norm, ddpm,
                    base_model, pocket_com_before, outfile_prefix, step_label):
    """
    Decode current ligand state into molecules and save snapshot SDF.
    """
    from analysis.molecule_builder import build_molecule, process_molecule

    try:
        x_lig_s, h_lig_s, x_pocket_s, _ = ddpm.sample_p_xh_given_z0(
            z_lig.clone(), xh_pocket.clone(),
            lig_mask, pocket_norm["mask"],
            (lig_mask.max().item() + 1))

        pocket_com_after_s = scatter_mean(x_pocket_s, pocket_norm["mask"], dim=0)
        shift_s = (pocket_com_before - pocket_com_after_s)[lig_mask]
        x_lig_s = x_lig_s + shift_s

        x_out_s  = x_lig_s.detach().cpu()
        atype_s  = h_lig_s.argmax(1).detach().cpu()
        mask_cpu = lig_mask.cpu()

        mols = []
        for pc in zip(generator_utils.batch_to_list(x_out_s, mask_cpu),
                      generator_utils.batch_to_list(atype_s, mask_cpu)):
            mol = build_molecule(*pc, base_model.dataset_info, add_coords=True)
            mol = process_molecule(mol, add_hydrogens=False, sanitize=True,
                                   relax_iter=0, largest_frag=True)
            if mol is not None:
                mols.append(mol)

        snap_path = f"{outfile_prefix}_step{step_label:04d}.sdf"
        generator_utils.write_sdf_file(snap_path, mols)
        logger.info("  [snapshot] step=%d | %d molecules -> %s",
                    step_label, len(mols), snap_path)

    except Exception as e:
        logger.warning("  [snapshot] step=%d failed: %s", step_label, e)


def guided_generate(
    base_ckpt, stage3_ckpt, pdbfile, npz_path,
    outfile, n_samples=10, guidance_scale=1.0,
    guide_every=10, timesteps=100, device="cuda",
    snapshot_steps=None):
    """
    Main guided generation pipeline.
    """
    dev = torch.device(device)

    outfile_stem = str(outfile).removesuffix(".sdf")
    if snapshot_steps is None:
        snapshot_steps = []

    logger.info("Loading base generator from %s", base_ckpt)
    base_model = LigandPocketDDPM.load_from_checkpoint(
        base_ckpt, map_location=dev)
    base_model = base_model.to(dev)
    base_model.eval()

    logger.info("Loading Stage-3 from %s", stage3_ckpt)
    stage3 = load_stage3(stage3_ckpt, dev)

    rec_feats, rec_centers, rec_mask, pocket_center = load_npz_pocket(
        npz_path, device=dev)

    from Bio.PDB import PDBParser
    pdb_struct = PDBParser(QUIET=True).get_structure("", pdbfile)[0]

    import glob
    sdf_candidates = glob.glob(
        f"/home/sirui/dataset/P-L/**/{os.path.basename(npz_path).replace('.npz', '')}_ligand.sdf",
        recursive=True)

    if sdf_candidates:
        ref_ligand = sdf_candidates[0]
        logger.info("Using ref ligand: %s", ref_ligand)
        residues = generator_utils.get_pocket_from_ligand(pdb_struct, ref_ligand)
    else:
        logger.warning("No ligand SDF found, using all residues as pocket")
        from Bio.PDB.Polypeptide import is_aa
        residues = [r for r in pdb_struct.get_residues() if is_aa(r, standard=True)]

    pocket = base_model.prepare_pocket(residues, repeats=n_samples)
    pocket_com_before = scatter_mean(pocket["x"], pocket["mask"], dim=0)

    num_nodes_lig = base_model.ddpm.size_distribution.sample_conditional(
        n1=None, n2=pocket["size"])

    ddpm = base_model.ddpm
    ddpm.eval()
    timesteps = timesteps or ddpm.T
    n_samples_actual = len(pocket["size"])

    _, pocket_norm = ddpm.normalize(pocket=pocket)
    xh0_pocket = torch.cat([pocket_norm["x"], pocket_norm["one_hot"]], dim=1)

    lig_mask = generator_utils.num_nodes_to_batch_mask(
        n_samples_actual, num_nodes_lig, dev)

    mu_lig_x = scatter_mean(pocket_norm["x"], pocket_norm["mask"], dim=0)
    mu_lig_h = torch.zeros((n_samples_actual, ddpm.atom_nf), device=dev)
    mu_lig   = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
    sigma    = torch.ones_like(pocket_norm["size"]).unsqueeze(1)

    z_lig, xh_pocket = ddpm.sample_normal_zero_com(
        mu_lig, xh0_pocket, sigma, lig_mask, pocket_norm["mask"])

    logger.info("Starting guided reverse diffusion (%d steps, guidance=%.2f, every=%d)",
                timesteps, guidance_scale, guide_every)

    if not snapshot_steps:
        n_snaps = 5
        snapshot_steps = [int(round(timesteps * i / (n_snaps - 1)))
                          for i in range(n_snaps)]
        snapshot_steps = list(set(max(0, min(timesteps - 1, s))
                                  for s in snapshot_steps))

    snapshot_set = set(snapshot_steps)
    logger.info("Snapshots will be saved at steps: %s", sorted(snapshot_set))

    for s in reversed(range(0, timesteps)):
        s_arr = torch.full((n_samples_actual, 1), fill_value=s, device=dev)
        t_arr = s_arr + 1
        s_arr = s_arr / timesteps
        t_arr = t_arr / timesteps

        with torch.no_grad():
            z_lig, xh_pocket = ddpm.sample_p_zs_given_zt(
                s_arr, t_arr, z_lig, xh_pocket, lig_mask, pocket_norm["mask"])

        if guidance_scale > 0 and s % guide_every == 0:
            try:
                grad = compute_guidance_grad(
                    stage3, rec_feats, rec_centers, rec_mask, pocket_center,
                    z_lig, lig_mask, n_samples_actual, guidance_scale)

                z_lig = z_lig.contiguous().clone()
                z_lig[:, :3] = z_lig[:, :3] + grad

                from torch_scatter import scatter_mean as sm
                com = sm(z_lig[:, :3], lig_mask, dim=0)[lig_mask]
                z_lig[:, :3] = z_lig[:, :3] - com

                if s % 50 == 0:
                    logger.info("  step %d | guidance applied | grad_norm=%.4f",
                                s, grad.norm().item())

            except Exception as e:
                logger.warning("  step %d | guidance failed: %s", s, e)

        if s in snapshot_set:
            decode_snapshot(z_lig, xh_pocket, lig_mask, pocket_norm, ddpm,
                            base_model, pocket_com_before, outfile_stem, s)

    x_lig, h_lig, x_pocket, h_pocket = ddpm.sample_p_xh_given_z0(
        z_lig, xh_pocket, lig_mask, pocket_norm["mask"], n_samples_actual)

    pocket_com_after = scatter_mean(
        x_pocket, pocket_norm["mask"], dim=0)
    shift = (pocket_com_before - pocket_com_after)[lig_mask]
    x_lig = x_lig + shift

    from analysis.molecule_builder import build_molecule, process_molecule

    x_out        = x_lig.detach().cpu()
    atom_type    = h_lig.argmax(1).detach().cpu()
    lig_mask_cpu = lig_mask.cpu()

    molecules = []
    for mol_pc in zip(generator_utils.batch_to_list(x_out, lig_mask_cpu),
                      generator_utils.batch_to_list(atom_type, lig_mask_cpu)):
        mol = build_molecule(*mol_pc, base_model.dataset_info, add_coords=True)
        mol = process_molecule(mol, add_hydrogens=False, sanitize=True,
                               relax_iter=0, largest_frag=True)
        if mol is not None:
            molecules.append(mol)

    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    generator_utils.write_sdf_file(outfile, molecules)
    logger.info("Saved %d molecules to %s", len(molecules), outfile)

    return molecules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Base generator + Stage-3 Gradient Inversion guided generation")

    parser.add_argument("--base_ckpt", type=str,
        default=str(Path(__file__).resolve().parent / "checkpoints" / "generator" / "model.ckpt"))
    parser.add_argument("--stage3_ckpt", type=str,
        default=str(Path(__file__).resolve().parent / "checkpoints" / "stage3" / "ckpt_final.pt"))
    parser.add_argument("--pdbfile", type=str, required=True)
    parser.add_argument("--npz", type=str, required=True,
        help="Stage-1 .npz file for the same protein")
    parser.add_argument("--outfile", type=str, default="./guided_output.sdf")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guide_every", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snapshot_steps", type=int, nargs="+", default=None,
        help="Timestep indices at which to save snapshot SDF files. Default: 5 evenly-spaced steps.")

    args = parser.parse_args()

    guided_generate(
        base_ckpt        = args.base_ckpt,
        stage3_ckpt      = args.stage3_ckpt,
        pdbfile          = args.pdbfile,
        npz_path         = args.npz,
        outfile          = args.outfile,
        n_samples        = args.n_samples,
        guidance_scale   = args.guidance_scale,
        guide_every      = args.guide_every,
        timesteps        = args.timesteps,
        device           = args.device,
        snapshot_steps   = args.snapshot_steps,
    )