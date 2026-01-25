# Quantifying the Explore-Exploit Trade-off in Metaheuristics via Normalized Positional Diversity

This repository contains the official implementation of the $D^*$ diversity metric, as introduced in our study. This tool provides a normalized, landscape-independent framework for monitoring the exploration and exploitation dynamics of metaheuristic optimizers in real-time.

## Project Structure
- **src/**: Core Python implementation of the $D^*$ metric and benchmark optimizers (e.g., AGOA, GWO, EES).
- **notebooks/**: Jupyter notebooks for generating diversity profiles, sensitivity analysis, and statistical plots.
- **requirements.txt**: List of necessary Python libraries to run the experiments.

## Installation
To set up the environment and ensure reproducibility, please install the required dependencies:
```bash
pip install -r requirements.txt

How to Reproduce Results
1. Running Experiments
To execute the optimizers on the CEC 2020 benchmark suite and collect diversity data, run:
python src/main_cec2020.py

2. Visualization and Analysis
For detailed analysis, including generating diversity profiles with Standard Error (SE) bands and correlation markers, refer to the following notebook:

notebooks/[finalized] updated_result.ipynb

3. Sensitivity Analysis
To verify the stability of the metric across different scales and parameters:


Practical UtilityThe _D*_ metric can be utilized to design adaptive search policies. Examples of its application include:Search Mode Switching: Transitioning between exploration and local search operators based on _D*_ thresholds.Parameter Adaptation: Dynamically adjusting mutation rates or selection pressure to counteract premature convergence.

Supplementary Materials
All raw data, statistical test results, and additional high-resolution plots are included in the notebooks/ directory to ensure full transparency of the reported results.





