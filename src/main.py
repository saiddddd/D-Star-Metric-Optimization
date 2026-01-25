import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from optimizers import *
from evaluator import (summarize_run, plot_profile, plot_comparison, 
                       plot_global_summary, plot_global_search_dynamics,
                       plot_exploration_fitness_scatter, generate_correlation_report)
from classical_benchmark_function_dim30 import fun_info as fun_info_30
from classical_benchmark_function_dim50 import fun_info as fun_info_50
from classical_benchmark_function_dim100 import fun_info as fun_info_100

try:
    from Get_Functions_details import Get_Functions_details
except ImportError:
    def Get_Functions_details(n):
        return -100, 100, 30, lambda x: np.sum(x**2)

def main():
    func_names = [
                    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13",
                #   "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23"
                  ] 
    
    optimizers = {"GWO": GWO_record, 
                  "PSO": PSO_record, 
                  "DE": DE_record, 
                  "AGOA": AGOA_record,
                  "EES": EES_record} 
    
    RUNS = 30  
    MAX_ITER = 500
    POP_SIZE = 30
    
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_summaries = []
    detailed_data = {}
    individual_runs_data = [] 

    for fname in func_names:
        lb, ub, dim, fobj = fun_info_100(fname)
        detailed_data[fname] = {}

        for opt_name, opt_func in optimizers.items():
            print(f"Executing {opt_name} on {fname}...")
            run_fitness, run_stats, profiles = [], [], []

            for r in range(RUNS):
                rec = opt_func(fobj, lb, ub, dim, POP_SIZE, MAX_ITER)
                st = summarize_run(rec)
                
                run_fitness.append(rec["best_f"])
                run_stats.append(st)
                profiles.append(rec)
                
                individual_runs_data.append({
                    "Function": fname,
                    "Optimizer": opt_name,
                    "Run": r + 1,
                    "Best_Fitness": rec["best_f"],
                    "DStar_Mean_Expl": st["DStar_Mean_Expl"],
                    "DStar_Mean_Exploit": st["DStar_Mean_Exploit"],
                    "DStar_AUC_Expl": st["DStar_AUC_Expl"],
                    "DStar_AUC_Exploit": st["DStar_AUC_Exploit"],
                    "DStar_Early_5": st["DStar_Early_5"],
                    "DStar_Early_10": st["DStar_Early_10"],
                    "DStar_Xover": st["DStar_Xover"],
                    "DStar_Conv": st["DStar_Conv"],
                    "Hussain_Mean_Expl": st["Hussain_Mean_Expl"],
                    "Hussain_Mean_Exploit": st["Hussain_Mean_Exploit"],
                    "Hussain_AUC_Expl": st["Hussain_AUC_Expl"],
                    "Hussain_AUC_Exploit": st["Hussain_AUC_Exploit"],
                    "Hussain_Early_5": st["Hussain_Early_5"],
                    "Hussain_Early_10": st["Hussain_Early_10"],
                    "Hussain_Xover": st["Hussain_Xover"],
                    "Hussain_Conv": st["Hussain_Conv"]
                })
            
            all_summaries.append({
                "Function": fname,
                "Optimizer": opt_name,
                "Mean_Fitness": np.mean(run_fitness),
                "Std_Fitness": np.std(run_fitness),
                "Mean_Expl": np.mean([s["DStar_Mean_Expl"] for s in run_stats]),
                "Mean_Exploit": np.mean([s["DStar_Mean_Exploit"] for s in run_stats]),
                "AUC_Expl": np.mean([s["DStar_AUC_Expl"] for s in run_stats]),
                "AUC_Exploit": np.mean([s["DStar_AUC_Exploit"] for s in run_stats]),
                "Mean_Expl_Hussain": np.mean([s["Hussain_Mean_Expl"] for s in run_stats]),
                "Mean_Exploit_Hussain": np.mean([s["Hussain_Mean_Exploit"] for s in run_stats]),
                "AUC_Expl_Hussain": np.mean([s["Hussain_AUC_Expl"] for s in run_stats]),
                "AUC_Exploit_Hussain": np.mean([s["Hussain_AUC_Exploit"] for s in run_stats]),
                "Avg_Conv_Iter": np.mean([s["DStar_Conv"] for s in run_stats]),
                "Avg_Xover_Point": np.mean([s["DStar_Xover"] for s in run_stats]),
                "Avg_Xover_Hussain": np.mean([s["Hussain_Xover"] for s in run_stats])
            })
            detailed_data[fname][opt_name] = profiles

            sample_title = f"{opt_name}_{fname}_Dynamics"
            plot_profile(profiles[0], title=f"{opt_name} Search Dynamics - {fname}")
            plt.savefig(os.path.join(output_dir, f"{sample_title}.png"), bbox_inches='tight', dpi=300)
            plt.close() 

        plot_comparison(detailed_data, fname, output_dir)

    df_ind = pd.DataFrame(individual_runs_data)
    df_ind.to_csv(os.path.join(output_dir, "individual_runs_history.csv"), index=False)

    df_res = pd.DataFrame(all_summaries)
    df_res.to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)

    plot_global_summary(detailed_data, output_dir, "Classical_100D")
    plot_global_search_dynamics(detailed_data, output_dir, "Classical_100D")

    plot_exploration_fitness_scatter(df_ind, output_dir)
    generate_correlation_report(df_ind, output_dir)
    
    print(f"\nSimulation complete. Results in: '{output_dir}'")

if __name__ == "__main__":
    main()