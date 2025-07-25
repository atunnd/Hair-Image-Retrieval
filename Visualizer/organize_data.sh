#!/bin/bash

# Data Organization Script for Hairstyle Retrieval Visualizer
# This script helps organize your existing data files into the new structure

echo "🔄 Organizing data files into new structure..."

# Create directory structure
mkdir -p data/hairstyle_retrieval/{dino,mae,siamim,simclr,simmim}
mkdir -p data/k-hairstyle/{dino,mae,siamim,simclr,simmim}

echo "📁 Created directory structure"

# Function to move files if they exist
move_if_exists() {
    if [ -f "$1" ]; then
        mv "$1" "$2"
        echo "✅ Moved $1 -> $2"
    else
        echo "⚠️  File not found: $1"
    fi
}

# Move regular hairstyle results
echo "📦 Moving regular hairstyle results..."
move_if_exists "data/dino_top100_results.json" "data/hairstyle_retrieval/dino/dino_10k.json"
move_if_exists "data/mae_top100_results.json" "data/hairstyle_retrieval/mae/mae_10k.json"
move_if_exists "data/siamim_top100_results.json" "data/hairstyle_retrieval/siamim/siamim_10k.json"
move_if_exists "data/simclr_top100_results.json" "data/hairstyle_retrieval/simclr/simclr_10k.json"
move_if_exists "data/simmim_top100_results.json" "data/hairstyle_retrieval/simmim/simmim_10k.json"

# Move Korean hairstyle results
echo "📦 Moving Korean hairstyle results..."
move_if_exists "data/dino_k_hairstyle_results.json" "data/k-hairstyle/dino/dino_k_hairstyle_10k.json"
move_if_exists "data/mae_k_hairstyle_results.json" "data/k-hairstyle/mae/mae_k_hairstyle_10k.json"
move_if_exists "data/siamim_k_hairstyle_results.json" "data/k-hairstyle/siamim/siamim_k_hairstyle_10k.json"
move_if_exists "data/simclr_k_hairstyle_results.json" "data/k-hairstyle/simclr/simclr_k_hairstyle_10k.json"
move_if_exists "data/simmim_k_hairstyle_results.json" "data/k-hairstyle/simmim/simmim_k_hairstyle_10k.json"

echo "✨ Data organization complete!"
echo ""
echo "📋 New structure:"
echo "data/"
echo "├── hairstyle_retrieval/"
echo "│   ├── dino/dino_10k.json"
echo "│   ├── mae/mae_10k.json"
echo "│   ├── siamim/siamim_10k.json"
echo "│   ├── simclr/simclr_10k.json"
echo "│   └── simmim/simmim_10k.json"
echo "├── k-hairstyle/"
echo "│   ├── dino/dino_k_hairstyle_10k.json"
echo "│   ├── mae/mae_k_hairstyle_10k.json"
echo "│   ├── siamim/siamim_k_hairstyle_10k.json"
echo "│   ├── simclr/simclr_k_hairstyle_10k.json"
echo "│   └── simmim/simmim_k_hairstyle_10k.json"
echo "├── hairstyle_retrieval_benchmark.json"
echo "└── korean_hairstyle_retrieval_benchmark.json"
echo ""
echo "🚀 You can now run the application with the new structure!"
