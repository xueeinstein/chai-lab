import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


def chai_calc_pae(out_dir, pep_asym_id=2):
    out_dir = Path(out_dir)
    scores = np.load(out_dir / 'confidence_scores.npz')
    n = scores['pae'].shape[1]
    asym_ids = scores['token_asym_ids'][0, :n]
    
    with open(out_dir / 'ranking_debug.json', 'r') as f:
        rank = json.load(f)
        
    rank0 = rank[0]['model_idx']
    pae = scores['pae'][rank0]
    pep_chain_idx = np.where(asym_ids == pep_asym_id)[0]
    rep_chain_idx = np.where(asym_ids != pep_asym_id)[0]
    inter_pae = ((pae[pep_chain_idx, :][:, rep_chain_idx] + pae[rep_chain_idx, :][:, pep_chain_idx].T) / 2.0).mean()
    return inter_pae


def calc_pep_plddt(out_dir, pep_asym_id=2):
    out_dir = Path(out_dir)
    scores = np.load(out_dir / 'confidence_scores.npz')
    n = scores['plddt'].shape[1]
    asym_ids = scores['token_asym_ids'][0, :n]

    with open(out_dir / 'ranking_debug.json', 'r') as f:
        rank = json.load(f)
        
    rank0 = rank[0]['model_idx']
    plddt = scores['plddt'][rank0, :n]
    pep_chain_idx = np.where(asym_ids == pep_asym_id)[0]
    return plddt[pep_chain_idx].mean()


def calc_ptm(out_dir):
    out_dir = Path(out_dir)
    with open(out_dir / 'ranking_debug.json', 'r') as f:
        rank = json.load(f)
        
    return rank[0]['complex_ptm']


def calc_iptm(out_dir):
    out_dir = Path(out_dir)
    with open(out_dir / 'ranking_debug.json', 'r') as f:
        rank = json.load(f)
        
    return rank[0]['interface_ptm']


def collect_scores(out_dir):
    scores = []
    out_dir = Path(out_dir)
    for sub_dir in out_dir.iterdir():
        if (sub_dir / 'confidence_scores.npz').exists():
            scores.append({
                'name': sub_dir.name,
                'inter_pae': chai_calc_pae(sub_dir),
                'pep_plddt': calc_pep_plddt(sub_dir),
                'ptm': calc_ptm(sub_dir),
                'iptm': calc_iptm(sub_dir),
            })

    return scores


if __name__ == "__main__":
    output_dir = Path(sys.argv[1])
    scores = collect_scores(output_dir)

    df = pd.DataFrame(scores)
    df.to_csv(output_dir / 'scores.csv', index=False)
