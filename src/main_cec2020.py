import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from optimizers import *
from evaluator import (summarize_run, plot_profile, plot_comparison, 
                       plot_global_summary, plot_global_search_dynamics,
                       plot_exploration_fitness_scatter, generate_correlation_report)
from opfunu.cec_based.cec2020 import *

def main():
    cec_functions = {
        "F1": F12020, "F2": F22020, "F3": F32020, "F4": F42020, "F5": F52020,
        "F6": F62020, "F7": F72020, "F8": F82020, "F9": F92020, "F10": F102020
    }
    
    optimizers = {
        "GWO": GWO_record, 
        "PSO": PSO_record, 
        "DE": DE_record, 
        "AGOA": AGOA_record,
        "EES": EES_record
    } 

    RUNS = 30  
    MAX_ITER = 2000
    POP_SIZE = 50
    DIM = 30  
    LB, UB = -100, 100 
    
    output_dir = "results_cec2020"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_summaries = []
    detailed_data = {}
    individual_runs_data = []

    for fname, f_class in cec_functions.items():
        obj_cec = f_class(ndim=DIM)
        fobj = obj_cec.evaluate  
        
        detailed_data[fname] = {}

        for opt_name, opt_func in optimizers.items():
            print(f"Executing {opt_name} on CEC 2020 {fname} (Dim={DIM})...")
            run_fitness, run_stats, profiles = [], [], []

            for r in range(RUNS):
                rec = opt_func(fobj, LB, UB, DIM, POP_SIZE, MAX_ITER)
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
                "Avg_Conv_DStar": np.mean([s["DStar_Conv"] for s in run_stats]),
                "Avg_Conv_Hussain": np.mean([s["Hussain_Conv"] for s in run_stats]),
                "Avg_Xover_DStar": np.mean([s["DStar_Xover"] for s in run_stats]),
                "Avg_Xover_Hussain": np.mean([s["Hussain_Xover"] for s in run_stats])
            })
            detailed_data[fname][opt_name] = profiles

            sample_title = f"{opt_name}_{fname}_Dynamics"
            plot_profile(profiles[0], title=f"{opt_name} Search Dynamics - {fname}")
            plt.savefig(os.path.join(output_dir, f"{sample_title}.png"), bbox_inches='tight', dpi=300)
            plt.close() 

        plot_comparison(detailed_data, fname, output_dir)

    df_ind = pd.DataFrame(individual_runs_data)
    df_ind.to_csv(os.path.join(output_dir, "cec2020_individual_runs_history.csv"), index=False)

    df_res = pd.DataFrame(all_summaries)
    df_res.to_csv(os.path.join(output_dir, "cec2020_summary.csv"), index=False)

    plot_global_summary(detailed_data, output_dir, "CEC2020")
    plot_global_search_dynamics(detailed_data, output_dir, "CEC2020")

    plot_exploration_fitness_scatter(df_ind, output_dir)
    generate_correlation_report(df_ind, output_dir)

    raw_fitness_data = []
    for fname in cec_functions.keys():
        for opt_name in optimizers.keys():
            fitness_values = [p["best_f"] for p in detailed_data[fname][opt_name]]
            raw_fitness_data.append({
                "Function": fname,
                "Optimizer": opt_name,
                "Fitness_History": fitness_values
            })
    
    df_raw = pd.DataFrame(raw_fitness_data)
    df_raw.to_json(os.path.join(output_dir, "cec2020_raw_fitness.json"), orient="records", indent=4)
    
    print(f"\nCEC 2020 Simulation complete. Folder: '{output_dir}'")

if __name__ == "__main__":
    main()