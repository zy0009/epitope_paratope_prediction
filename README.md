# Bidirectional Cross-Modal Attention with Contrastive Learning for Epitope and Paratope Prediction

This repository contains the implementation of a multimodal residue-level framework for epitope and paratope prediction. The model integrates sequence-derived contextual representations and structure-derived graph representations through bidirectional cross-modal attention and cluster-aware contrastive learning.

## Overview

Accurate identification of antigen epitopes and antibody paratopes is important for vaccine design and therapeutic antibody engineering. Sequence-only methods may miss spatially proximal residues, while structure-only methods can underuse evolutionary and contextual sequence information. This project addresses these limitations by jointly modeling protein sequence context and residue-level structural neighborhoods.

The proposed framework consists of four main components:

- **Hybrid Sequence Attention Encoder (HSAE)**: refines ESM-derived residue features with BiLSTM, Transformer self-attention, and hybrid pooling.
- **Node-Edge Attention Graph Encoder (NEAGE)**: models residue-level structural graphs using node and edge features.
- **Cross-Modal Interaction and Alignment (CMIA)**: exchanges information between sequence and structure representations using bidirectional cross-modal attention.
- **Cluster-aware contrastive learning**: aligns sequence and structural embeddings while reducing noisy negative effects.

## Model Architecture

![Architecture of the proposed framework](figures/architecture.png)

## Visualization and Analysis

### t-SNE Visualization

The learned fused residue representations are visualized for epitope and paratope prediction. Red points denote binding residues, and blue-gray points denote non-binding residues.

![t-SNE visualization](figures/tsne.png)

### Ablation Study

The ablation results show the performance degradation of model variants relative to the full model.

![Ablation study](figures/ablation.png)

### SARS-CoV-2 Case Study

The case study evaluates prediction performance across five SARS-CoV-2 B-cell epitope types and visualizes representative antigen-antibody binding interfaces.

![SARS-CoV-2 case study performance](figures/case_study_performance.png)

![Binding visualization](figures/case_study_visualization.png)

## Repository Structure

```text
.
├── config.py      # Hyperparameter and path configuration
├── layers.py      # HSAE, NEAGE, and neural network layers
├── losses.py      # Weighted BCE and contrastive learning losses
├── metrics.py     # Evaluation metrics
├── models.py      # Full prediction model
├── train.py       # Training and cross-validation script
├── test.py        # Testing script
├── utils.py       # Data loading and utility functions
└── figures/       # Figures used in this README
```

## Data
Datasets can be downloaded from zenodo here: https://zenodo.org/record/3885236.

The code expects preprocessed graph datasets in pickle format. By default, paths are configured in `config.py`:

```python
train_dataset_path = './datasets/train.cpkl'
val_dataset_path = './datasets/val.cpkl'
test_dataset_path = './datasets/test.cpkl'
```

Each protein graph contains residue features, edge features, neighborhood indices, and residue-level labels.

## Usage

Install the required Python packages according to your local PyTorch/CUDA environment. The main dependencies include:

```text
python
numpy
scikit-learn
torch
```

Train the model:

```bash
python train.py
```

Evaluate the model:

```bash
python test.py
```

Before running, update dataset paths, GPU settings, and output paths in `config.py` if needed.

## Main Results

Under the adopted data splits and five-fold cross-validation setting, the proposed method achieves:

- Epitope prediction: MCC = 0.430
- Paratope prediction: MCC = 0.661

Additional evaluation includes a large-scale independent AsEP benchmark, ablation studies, network-depth analysis, t-SNE visualization, and a SARS-CoV-2 case study.
