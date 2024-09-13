import json
import numpy as np
from pathlib import Path


def save_ranking_debug(output_pdb_paths, ranking_data, output_dir):
    ranking_scores = []
    for idx, (pdb_path, ranking) in enumerate(zip(output_pdb_paths, ranking_data)):
        score = {
            "model_idx": idx,
            "pdb_path": pdb_path.name,
            "aggregate_score": ranking.aggregate_score.item(),
            "complex_ptm": ranking.ptm_scores.complex_ptm.item(),
            "interface_ptm": ranking.ptm_scores.interface_ptm.item(),
            "complex_plddt": ranking.plddt_scores.complex_plddt.item(),
            "total_clashes": ranking.clash_scores.total_clashes.item(),
            "has_clashes": ranking.clash_scores.has_clashes.item(),
        }
        ranking_scores.append(score)
    
    # Sort the ranking scores by aggregate_score in descending order
    ranking_scores.sort(key=lambda x: x["aggregate_score"], reverse=True)
    
    # Save the sorted ranking scores as JSON
    output_path = Path(output_dir) / "ranking_debug.json"
    with output_path.open("w") as f:
        json.dump(ranking_scores, f, indent=2)
    
    print(f"Saved ranking debug information to {output_path}")


def save_confidence_scores(confidence_scores, token_asym_ids, output_dir):
    output_path = Path(output_dir) / "confidence_scores.npz"
    
    np.savez(
        output_path,
        pae=confidence_scores.pae.cpu().numpy(),
        pde=confidence_scores.pde.cpu().numpy(),
        plddt=confidence_scores.plddt.cpu().numpy(),
        token_asym_ids=token_asym_ids.cpu().numpy(),
    )
    
    print(f"Saved confidence scores to {output_path}")