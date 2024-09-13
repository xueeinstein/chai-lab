import os
import sys
from pathlib import Path

# Add project root folder to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(proj_root)

import numpy as np
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

from chai_lab.chai1 import run_inference
from chai_lab.utils.save_scores import save_ranking_debug, save_confidence_scores

# We use fasta-like format for inputs.
# Every record may encode protein, ligand, RNA or DNA
#  see example below

example_fasta = """
>protein|example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|example-of-peptide
GAAL
>ligand|and-example-for-ligand-encoded-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()

if len(sys.argv) != 3 and len(sys.argv) != 1:
    print(f"Usage: {sys.argv[0]} <fasta_path> <output_dir>")
    sys.exit(1)

if len(sys.argv) == 3:
    fasta_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
else:
    fasta_path = Path("/tmp/example.fasta")
    output_dir = Path("/tmp/outputs")
    fasta_path.write_text(example_fasta)

inputs_features, output_pdb_paths, confidence_scores, ranking_data, msa_plot_path = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # 'default' setup
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device=torch.device("cuda:0"),
    use_esm_embeddings=True,
)
save_ranking_debug(output_pdb_paths, ranking_data, output_dir)
save_confidence_scores(confidence_scores, inputs_features["token_asym_id"], output_dir)

# Load pTM, ipTM, pLDDTs and clash scores for sample 2
# scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))
