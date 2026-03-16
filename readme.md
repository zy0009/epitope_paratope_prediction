# A Unified Framework for Epitope and Paratope Prediction via Multimodal Contrastive Learning
## Abstract
Accurate prediction of antigenic epitopes and antibody paratopes is crucial for advancing vaccine design and therapeutic antibody development. However, sequence-based methods are limited by their inability to model long-range interactions, while structure-only models neglect essential evolutionary information. Moreover, many current multimodal models rely on simplistic fusion strategies that yield suboptimal alignment between sequence and structure, ultimately hindering predictive accuracy. To overcome these limitations, we propose a unified multimodal contrastive learning framework for accurate epitope and paratope prediction. Our model comprises two core components: a Hybrid Sequence Attention Encoder (HSAE) that processes ESM-3-derived embeddings through Transformer self-attention and gated hybrid pooling to capture long-range dependencies and global sequence context; and a Node-Edge Attention Graph Encoder (NEAGE) that operates on residue-level graphs constructed from experimentally resolved structures, using edge-aware attention aggregation to model local geometric and relational features. To explicitly align these two distinct modalities, we introduce a residue-level cross-modal contrastive learning objective that maps sequence and structural embeddings into a shared latent space, yielding discriminative and comprehensive residue embeddings. Extensive experiments on two benchmark datasets demonstrate that our method consistently outperforms state-of-the-art baselines across all evaluation metrics for both epitope and paratope prediction. Moreover, a case study on SARS-CoV-2 B-cell epitopes demonstrates that our model captures antigen–antibody interaction patterns more precisely, underscoring its potential for real-world applications in immunology and drug discovery. Our code and models are publicly available at https://github.com/zy0009/epitope_paratope_prediction.

## Datasets
Datasets can be downloaded from zenodo here: https://zenodo.org/record/3885236.

The data format is the following:
- list of dictionaries, each corresponding to a protein
- - Each dictionary has the following entries(keys):
- - - complex_code: str
- - - l_vertex: numpy matrix of residues in primary protein (dimensions: num_residues x 63)
- - - - Indices [0,20] represent one hot encoding of the residue. Last entry for unresolved amino acid '*'.
- - - - Indices [21,40] represent PSSM entry for the
- - - - Indices [41:42] represent solvent accessibility
- - - - Indices [43:62] represent neighborhood composition 
- - - l_edge: numpy matrix of edges, currently empty as we don't consider edge features (dimensions: num_residues x 25 x 2)
- - - l_hood_indices: numpy matrix specifying spatial neighborhood for graph convolution (dimensions: num_residues x 25)
- - - label: numpy matrix with labels {1: interface, -1: non-interface} (dimensions: num_residues x 2)
- - - r_vertex: same as l_vertex, but for secondary protein
- - - r_edge: same as l_edge, but for secondary protein
- - - r_hood_indices: same as l_hood_indices, but for secondary protein
- - - label_r: same as label, but for secondary protein

## Requirement
- Python = 3.9.10
- Pytorch = 1.10.2
- Scikit-learn = 1.0.2

## Usage
All configurations are defined in `config.py`.

To train the model, run:

```bash
python train.py
```
To evaluate the model, run:
```bash
python test.py
```