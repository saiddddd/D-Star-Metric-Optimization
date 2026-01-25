import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon, friedmanchisquare, spearmanr
from itertools import combinations

def run_statistical_test(input_csv="results/individual_runs_history.csv"):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Please run the simulation first.")
        return

    df = pd.read_csv(input_csv)
    functions = df['Function'].unique()
    optimizers = df['Optimizer'].unique()
    output_path = "results"
    
    rank_data = []
    friedman_results = []

    for fname in functions:
        f_data = df[df['Function'] == fname]
        fitness_list = []
        temp_ranks = {}
        
        for opt in optimizers:
            fits = f_data[f_data['Optimizer'] == opt]['Best_Fitness'].values
            fitness_list.append(fits)
            temp_ranks[opt] = np.mean(fits)
        
        try:
            stat, p_f = friedmanchisquare(*fitness_list)
            friedman_results.append({"Function": fname, "Friedman_P_Value": f"{p_f:.6e}"})
        except:
            friedman_results.append({"Function": fname, "Friedman_P_Value": "Tie/Error"})
        
        sorted_opts = sorted(temp_ranks, key=temp_ranks.get)
        for rank, opt in enumerate(sorted_opts, 1):
            rank_data.append({"Function": fname, "Optimizer": opt, "Rank": rank})

    df_rank = pd.DataFrame(rank_data)
    mean_ranks = df_rank.groupby("Optimizer")["Rank"].mean().sort_values()

    wilcoxon_fitness = []
    for fname in functions:
        f_data = df[df['Function'] == fname]
        for opt1, opt2 in combinations(optimizers, 2):
            try:
                fits1 = f_data[f_data['Optimizer'] == opt1]['Best_Fitness'].values
                fits2 = f_data[f_data['Optimizer'] == opt2]['Best_Fitness'].values
                
                if np.array_equal(fits1, fits2):
                    p_w = 1.0
                else:
                    _, p_w = wilcoxon(fits1, fits2)
                
                wilcoxon_fitness.append({
                    "Function": fname, "Comparison": f"{opt1} vs {opt2}",
                    "P-Value": f"{p_w:.6e}", "Significant": "Yes" if p_w < 0.05 else "No"
                })
            except:
                continue

    stability_results = []
    for opt in optimizers:
        opt_data = df[df['Optimizer'] == opt]
        cv_star_list, cv_hussain_list = [], []
        
        for fname in functions:
            f_runs = opt_data[opt_data['Function'] == fname]
            cv_star = np.std(f_runs['DStar_Mean_Expl']) / (np.mean(f_runs['DStar_Mean_Expl']) + 1e-12)
            cv_h = np.std(f_runs['Hussain_Mean_Expl']) / (np.mean(f_runs['Hussain_Mean_Expl']) + 1e-12)
            cv_star_list.append(cv_star)
            cv_hussain_list.append(cv_h)
        
        try:
            _, p_cv = wilcoxon(cv_star_list, cv_hussain_list)
        except:
            p_cv = 1.0
            
        stability_results.append({
            "Optimizer": opt,
            "Mean_CV_DStar": np.mean(cv_star_list),
            "Mean_CV_Hussain": np.mean(cv_hussain_list),
            "P-Value_Stability": f"{p_cv:.6e}",
            "DStar_More_Stable": "Yes" if np.mean(cv_star_list) < np.mean(cv_hussain_list) and p_cv < 0.05 else "No"
        })

    redundancy_results = []
    for opt in optimizers:
        opt_data = df[df['Optimizer'] == opt]
        corr, p_corr = spearmanr(opt_data['DStar_Early_5'], opt_data['Hussain_Early_5'])
        redundancy_results.append({
            "Optimizer": opt,
            "Spearman_Rho": corr,
            "P-Value_Redundancy": p_corr,
            "Non_Redundant": "Yes" if corr < 0.95 else "No"
        })

    print("\n--- Friedman Mean Ranking (Fitness) ---")
    print(mean_ranks)
    print("\n--- Metric Stability Analysis (D* vs Hussain CV) ---")
    print(pd.DataFrame(stability_results))
    print("\n--- Redundancy Analysis (Early 5%) ---")
    print(pd.DataFrame(redundancy_results))

    df_rank.to_csv(os.path.join(output_path, "friedman_ranks.csv"), index=False)
    pd.DataFrame(wilcoxon_fitness).to_csv(os.path.join(output_path, "wilcoxon_fitness_results.csv"), index=False)
    pd.DataFrame(stability_results).to_csv(os.path.join(output_path, "stability_analysis.csv"), index=False)
    pd.DataFrame(redundancy_results).to_csv(os.path.join(output_path, "redundancy_analysis.csv"), index=False)

if __name__ == "__main__":
    run_statistical_test()