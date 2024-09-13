#!/bin/bash
GPU_ID=4
FASTA_DIR=./affinity_test_in_chai1
OUTPUT_DIR=./affinity_test_out_chai1

for fasta_file in $FASTA_DIR/*.fasta; do
    output_dir=$OUTPUT_DIR/$(basename $fasta_file .fasta)
    echo ">>> Processing $fasta_file -> $output_dir"
    CUDA_VISIBLE_DEVICES=$GPU_ID python examples/predict_structure.py $fasta_file $output_dir
done