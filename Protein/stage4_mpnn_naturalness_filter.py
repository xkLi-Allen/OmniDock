# -*- coding: utf-8 -*-
"""ProteinMPNN sequence naturalness filter for Stage4 outputs."""
import argparse
import csv
import re
from pathlib import Path


def parse_score(header):
    m = re.search(r"score=([0-9.\-]+)", header)
    return float(m.group(1)) if m else float('inf')


def natural_metrics(seq):
    L = len(seq)
    counts = {aa: seq.count(aa) for aa in set(seq)}
    frac = {aa: c / L for aa, c in counts.items()}
    return {
        'length': L,
        'gly_frac': seq.count('G') / L,
        'pro_frac': seq.count('P') / L,
        'gp_frac': (seq.count('G') + seq.count('P')) / L,
        'max_aa_frac': max(frac.values()),
        'hydrophobic_frac': sum(seq.count(a) for a in 'AILVFMYW') / L,
        'charged_frac': sum(seq.count(a) for a in 'DEKR') / L,
        'polar_frac': sum(seq.count(a) for a in 'STNQH') / L,
        'aromatic_frac': sum(seq.count(a) for a in 'FYW') / L,
    }


def passes(m, args):
    return (
        m['gly_frac'] <= args.max_gly and
        m['pro_frac'] <= args.max_pro and
        m['gp_frac'] <= args.max_gp and
        m['max_aa_frac'] <= args.max_single_aa and
        m['hydrophobic_frac'] >= args.min_hydrophobic and
        m['charged_frac'] <= args.max_charged
    )


def natural_penalty(m, args):
    return (
        max(0.0, m['gly_frac'] - 0.08) * 5.0 +
        max(0.0, m['pro_frac'] - 0.06) * 5.0 +
        max(0.0, m['gp_frac'] - 0.14) * 4.0 +
        max(0.0, m['max_aa_frac'] - 0.20) * 3.0 +
        max(0.0, args.min_hydrophobic - m['hydrophobic_frac']) * 4.0 +
        max(0.0, m['charged_frac'] - 0.35) * 2.0
    )


def iter_fasta_records(path):
    lines = Path(path).read_text().splitlines()
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            continue
        header, seq = lines[i], lines[i + 1].strip()
        if header.startswith('>T='):
            yield header, seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fa_root', required=True, help='Directory containing ProteinMPNN .fa outputs')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--max_gly', type=float, default=0.18)
    ap.add_argument('--max_pro', type=float, default=0.12)
    ap.add_argument('--max_gp', type=float, default=0.25)
    ap.add_argument('--max_single_aa', type=float, default=0.35)
    ap.add_argument('--min_hydrophobic', type=float, default=0.20)
    ap.add_argument('--max_charged', type=float, default=0.45)
    args = ap.parse_args()

    rows = []
    for fa in sorted(Path(args.fa_root).glob('**/seqs/*.fa')):
        for header, seq in iter_fasta_records(fa):
            m = natural_metrics(seq)
            ok = passes(m, args)
            mpnn_score = parse_score(header)
            penalty = natural_penalty(m, args)
            rows.append({
                'pass': int(ok),
                'combined_score': mpnn_score + penalty,
                'mpnn_score': mpnn_score,
                'natural_penalty': penalty,
                'fasta': str(fa),
                'header': header,
                'sequence': seq,
                **m,
            })

    rows.sort(key=lambda r: (1 - r['pass'], r['combined_score']))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        fieldnames = ['pass', 'combined_score', 'mpnn_score', 'natural_penalty', 'fasta', 'header', 'sequence',
                      'length', 'gly_frac', 'pro_frac', 'gp_frac', 'max_aa_frac', 'hydrophobic_frac',
                      'charged_frac', 'polar_frac', 'aromatic_frac']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(r['pass'] for r in rows)
    print(f'[Naturalness] records={len(rows)} pass={n_pass} out={args.out_csv}')
    if rows:
        best = rows[0]
        print(f"[Best] pass={best['pass']} combined={best['combined_score']:.3f} seq={best['sequence']}")


if __name__ == '__main__':
    main()
