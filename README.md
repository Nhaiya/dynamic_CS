# Brain Network Analysis Using Graph Mining

Extends the static contrast subgraph method (Lanciano et al., 2020) to dynamic
brain networks with two variants: Temporal and Multi-Objective.

## Structure
- `Core/` — original contrast subgraph algorithm (Lanciano et al.)
- `Variants/` — proposed Temporal and Multi-Objective variants
- `Datasets/` — derived networks (Schizophrenia, Parkinson, Multiple Sclerosis)
- `Results/` — classification and biomarker results

## Requirements
Python (`numpy`, `scipy`, `networkx`) and MATLAB with CVX (SeDuMi).

## Usage
Set the data paths at the top of a script and run:

python Variants/Tempral.py
python Variants/Multi_objective.py

## Data
Networks are derived from public datasets and subject to their original licenses
(COBRE, Neurocon, and the Multiple Sclerosis datasets). Alzheimer's (OASIS-3)
data cannot be redistributed under its Data Use Agreement and is not included;
request access at https://sites.wustl.edu/oasisbrains/.

## Reference
Lanciano T, Bonchi F, Gionis A. Explainable classification of brain networks via contrast subgraphs. InProceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 2020 Aug 23 (pp. 3308-3318).
