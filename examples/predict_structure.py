import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root folder to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(proj_root)

import numpy as np
import torch
from tqdm import tqdm

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

from chai_lab.chai1 import (
    load_exported,
    run_folding_on_context,
    read_inputs,
    load_chains_from_raw,
    AllAtomStructureContext,
    MSAContext,
    TemplateContext,
    get_esm_embedding_context,
    EmbeddingContext,
    ConstraintContext,
    AllAtomFeatureContext,
    MAX_MSA_DEPTH,
    MAX_NUM_TEMPLATES,
    AVAILABLE_MODEL_SIZES,
)
from chai_lab.utils.save_scores import save_ranking_debug, save_confidence_scores

# We use fasta-like format for inputs.
# Every record may encode protein, ligand, RNA or DNA
#  see example below

# Require type hinting, e.g. `>protein|xxx` or `>ligand|xxx`
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

def load_models(n_tokens: int, device: torch.device):
    """Load all required models for inference."""
    model_size = min(x for x in AVAILABLE_MODEL_SIZES if n_tokens <= x)
    
    feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
    token_input_embedder = load_exported(f"{model_size}/token_input_embedder.pt2", device)
    trunk = load_exported(f"{model_size}/trunk.pt2", device)
    diffusion_module = load_exported(f"{model_size}/diffusion_module.pt2", device)
    confidence_head = load_exported(f"{model_size}/confidence_head.pt2", device)

    return {
        'feature_embedding': feature_embedding,
        'token_input_embedder': token_input_embedder,
        'trunk': trunk,
        'diffusion_module': diffusion_module,
        'confidence_head': confidence_head
    }

def process_single_fasta(
    fasta_path: Path,
    output_dir: Path,
    device: torch.device,
    models: Optional[dict] = None,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    use_esm_embeddings: bool = True,
) -> Tuple[Path, List[Path]]:
    """Process a single FASTA file, optionally using pre-loaded models."""
    
    # Create output subdirectory named after the FASTA file
    fasta_output_dir = output_dir / fasta_path.stem
    fasta_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read inputs and prepare feature context
    fasta_inputs = read_inputs(fasta_path, length_limit=None)
    chains = load_chains_from_raw(fasta_inputs)
    contexts = [c.structure_context for c in chains]
    merged_context = AllAtomStructureContext.merge(contexts)
    n_actual_tokens = merged_context.num_tokens
    
    # Load or reuse models based on sequence length
    if models is None:
        models = load_models(n_actual_tokens, device)
    
    # Prepare remaining contexts
    msa_context = MSAContext.create_empty(n_tokens=n_actual_tokens, depth=MAX_MSA_DEPTH)
    main_msa_context = MSAContext.create_empty(n_tokens=n_actual_tokens, depth=MAX_MSA_DEPTH)
    template_context = TemplateContext.empty(n_tokens=n_actual_tokens, n_templates=MAX_NUM_TEMPLATES)
    
    if use_esm_embeddings:
        embedding_context = get_esm_embedding_context(chains, device=device)
    else:
        embedding_context = EmbeddingContext.empty(n_tokens=n_actual_tokens)
    
    constraint_context = ConstraintContext.empty()
    
    # Build final feature context
    feature_context = AllAtomFeatureContext(
        chains=chains,
        structure_context=merged_context,
        msa_context=msa_context,
        main_msa_context=main_msa_context,
        template_context=template_context,
        embedding_context=embedding_context,
        constraint_context=constraint_context,
    )
    
    # Run folding with pre-loaded models
    inputs, output_paths, confidence_scores, ranking_data, msa_plot_path = run_folding_on_context(
        feature_context=feature_context,
        output_dir=fasta_output_dir,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        seed=seed,
        device=device,
        models=models,  # Pass pre-loaded models
    )

    # Save results
    save_ranking_debug(output_paths, ranking_data, fasta_output_dir)
    save_confidence_scores(confidence_scores, inputs["token_asym_id"], fasta_output_dir)

    return fasta_output_dir, output_paths

def main():
    if len(sys.argv) not in [2, 3]:
        print(f"Usage: {sys.argv[0]} <fasta_path_or_dir> [output_dir]")
        print("  fasta_path_or_dir: Path to a FASTA file or directory containing FASTA files")
        print("  output_dir: Optional output directory (default: /tmp/outputs)")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("/tmp/outputs")

    # Collect all FASTA files to process
    if input_path.is_file():
        fasta_files = [input_path]
    else:
        fasta_files = list(input_path.glob("*.fasta")) + list(input_path.glob("*.fa"))
        if not fasta_files:
            print(f"No FASTA files found in directory: {input_path}")
            sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = None  # Will be loaded based on first sequence length

    # Process each FASTA file
    for fasta_path in tqdm(fasta_files, desc="Processing FASTA files"):
        try:
            fasta_output_dir, output_paths = process_single_fasta(
                fasta_path=fasta_path,
                output_dir=output_dir,
                device=device,
                models=models
            )
            print(f"Processed {fasta_path.name} -> {fasta_output_dir}")
            
            # Load models if not already loaded
            # FIXME: when n_tokens becomes larger, we need to load models dynamically
            if models is None:
                # Read first FASTA to determine sequence length
                fasta_inputs = read_inputs(fasta_path, length_limit=None)
                chains = load_chains_from_raw(fasta_inputs)
                contexts = [c.structure_context for c in chains]
                merged_context = AllAtomStructureContext.merge(contexts)
                n_tokens = merged_context.num_tokens
                models = load_models(n_tokens, device)
                
        except Exception as e:
            print(f"Error processing {fasta_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
