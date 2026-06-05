# -*- coding: utf-8 -*-
"""Score generated ligand backbone against native receptor-ligand interface geometry."""
import argparse
import csv
from pathlib import Path
import numpy as np


def read_chain_ca(pdb_path, chains):
    chains=set(chains)
    coords=[]
    resids=[]
    for line in Path(pdb_path).read_text(errors='ignore').splitlines():
        if not line.startswith('ATOM'):
            continue
        if line[12:16].strip() != 'CA':
            continue
        ch=line[21].strip()
        if ch not in chains:
            continue
        xyz=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
        coords.append(xyz)
        resids.append((ch, int(line[22:26])))
    if not coords:
        raise ValueError(f'No CA atoms found for chains {chains} in {pdb_path}')
    return np.stack(coords, axis=0), resids


def read_generated_ca(pdb_path):
    coords=[]
    for line in Path(pdb_path).read_text(errors='ignore').splitlines():
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f'No generated CA atoms in {pdb_path}')
    return np.asarray(coords, dtype=np.float32)


def rg(ca):
    c = ca.mean(axis=0, keepdims=True)
    return float(np.sqrt(((ca - c) ** 2).sum(axis=1).mean()))


def kabsch_align(pred, target):
    pc = pred.mean(axis=0, keepdims=True)
    qc = target.mean(axis=0, keepdims=True)
    p0 = pred - pc
    q0 = target - qc
    h = p0.T @ q0
    u, _, vh = np.linalg.svd(h)
    r = vh.T @ u.T
    if np.linalg.det(r) < 0:
        vh[-1, :] *= -1
        r = vh.T @ u.T
    return (pred - pc) @ r + qc


def kabsch_shape_metrics(pred_ca, target_ca):
    n_cmp = min(len(pred_ca), len(target_ca))
    pred_cmp = pred_ca[:n_cmp]
    target_cmp = target_ca[:n_cmp]
    aligned = kabsch_align(pred_cmp, target_cmp)
    aligned_rmsd = float(np.sqrt(((aligned - target_cmp) ** 2).sum(axis=1).mean()))
    dist_error = float(np.mean(np.abs(np.linalg.norm(aligned[:, None, :] - aligned[None, :, :], axis=-1) -
                                       np.linalg.norm(target_cmp[:, None, :] - target_cmp[None, :, :], axis=-1))))
    return aligned_rmsd, dist_error, n_cmp


def ca_ca_stats(ca):
    if len(ca) < 2:
        return 0.0, 0.0
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    return float(d.mean()), float(d.std())


def ca_clash_count(ca, min_dist=2.8):
    n_clashes = 0
    for i in range(len(ca)):
        for j in range(i + 3, len(ca)):
            if np.linalg.norm(ca[i] - ca[j]) < min_dist:
                n_clashes += 1
    return int(n_clashes)


def interface_subset(native_lig_ca, rec_ca, contact_cutoff):
    d = np.linalg.norm(native_lig_ca[:, None, :] - rec_ca[None, :, :], axis=-1)
    mask = d.min(axis=1) <= contact_cutoff
    if mask.sum() < 3:
        mask = np.ones(len(native_lig_ca), dtype=bool)
    return native_lig_ca[mask], mask


def score_one(gen_pdb, native_pdb, receptor_chains, ligand_chains, contact_cutoff=10.0, overlap_cutoff=8.0):
    rec_ca, _ = read_chain_ca(native_pdb, receptor_chains)
    lig_ca, _ = read_chain_ca(native_pdb, ligand_chains)
    gen_ca = read_generated_ca(gen_pdb)
    native_iface, native_mask = interface_subset(lig_ca, rec_ca, contact_cutoff)

    d_gen_rec = np.linalg.norm(gen_ca[:, None, :] - rec_ca[None, :, :], axis=-1).min(axis=1)
    gen_contact_frac = float((d_gen_rec <= contact_cutoff).mean())

    d_gen_native = np.linalg.norm(gen_ca[:, None, :] - native_iface[None, :, :], axis=-1)
    gen_native_overlap = float((d_gen_native.min(axis=1) <= overlap_cutoff).mean())
    native_covered = float((np.linalg.norm(native_iface[:, None, :] - gen_ca[None, :, :], axis=-1).min(axis=1) <= overlap_cutoff).mean())

    interface_aligned_rmsd, interface_dist_error, interface_cmp_size = kabsch_shape_metrics(gen_ca, native_iface)
    full_aligned_rmsd, full_dist_error, full_cmp_size = kabsch_shape_metrics(gen_ca, lig_ca)
    gen_ca_ca_mean, gen_ca_ca_std = ca_ca_stats(gen_ca)
    gen_clash_count = ca_clash_count(gen_ca)

    aligned_rmsd = interface_aligned_rmsd
    dist_error = interface_dist_error
    interface_distance = float(np.linalg.norm(gen_ca.mean(axis=0) - native_iface.mean(axis=0)))

    gen_centroid = gen_ca.mean(axis=0)
    native_centroid = native_iface.mean(axis=0)
    centroid_dist = float(np.linalg.norm(gen_centroid - native_centroid))
    native_rg = rg(native_iface)
    native_full_rg = rg(lig_ca)
    gen_rg = rg(gen_ca)
    rg_ratio = float(gen_rg / (native_rg + 1e-6))
    full_rg_ratio = float(gen_rg / (native_full_rg + 1e-6))

    score = (
        0.25 * gen_contact_frac +
        0.25 * gen_native_overlap +
        0.20 * native_covered +
        0.10 * np.exp(-centroid_dist / 20.0) -
        0.10 * abs(np.log(max(rg_ratio, 1e-6))) -
        0.10 * np.tanh(aligned_rmsd / 10.0)
    )
    return {
        'generated_pdb': str(gen_pdb),
        'native_pdb': str(native_pdb),
        'receptor_chains': receptor_chains,
        'ligand_chains': ligand_chains,
        'score': float(score),
        'gen_contact_frac': gen_contact_frac,
        'gen_native_overlap': gen_native_overlap,
        'contact_overlap': gen_native_overlap,
        'native_interface_covered': native_covered,
        'aligned_rmsd': aligned_rmsd,
        'dist_error': dist_error,
        'interface_distance': interface_distance,
        'centroid_dist': centroid_dist,
        'gen_rg': gen_rg,
        'native_interface_rg': native_rg,
        'rg_ratio': rg_ratio,
        'native_interface_size': int(len(native_iface)),
        'interface_subset_aligned_rmsd': interface_aligned_rmsd,
        'interface_subset_dist_error': interface_dist_error,
        'interface_subset_rg_ratio': rg_ratio,
        'interface_subset_cmp_size': int(interface_cmp_size),
        'full_aligned_rmsd': full_aligned_rmsd,
        'full_dist_error': full_dist_error,
        'full_rg_ratio': full_rg_ratio,
        'full_cmp_size': int(full_cmp_size),
        'native_full_rg': native_full_rg,
        'native_full_size': int(len(lig_ca)),
        'gen_ca_ca_mean': gen_ca_ca_mean,
        'gen_ca_ca_std': gen_ca_ca_std,
        'gen_clash_count': gen_clash_count,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--generated_pdb', required=True)
    ap.add_argument('--native_pdb', required=True)
    ap.add_argument('--receptor_chains', required=True)
    ap.add_argument('--ligand_chains', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--contact_cutoff', type=float, default=10.0)
    ap.add_argument('--overlap_cutoff', type=float, default=8.0)
    args=ap.parse_args()
    pdbs=sorted(Path(args.generated_pdb).glob('*.pdb')) if Path(args.generated_pdb).is_dir() else [Path(args.generated_pdb)]
    rows=[score_one(p, args.native_pdb, args.receptor_chains, args.ligand_chains, args.contact_cutoff, args.overlap_cutoff) for p in pdbs]
    rows.sort(key=lambda r:r['score'], reverse=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    best=rows[0]
    print(f"[NativeSimilarity] n={len(rows)} best_score={best['score']:.3f} contact={best['gen_contact_frac']:.3f} overlap={best['gen_native_overlap']:.3f} covered={best['native_interface_covered']:.3f} centroid_dist={best['centroid_dist']:.2f} interface_rmsd={best['interface_subset_aligned_rmsd']:.2f} full_rmsd={best['full_aligned_rmsd']:.2f}")
    print('[Best]', best['generated_pdb'])


if __name__=='__main__':
    main()
