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