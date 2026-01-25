import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def hussain_diversity(X):
    X = np.asarray(X)
    N, D = X.shape
    medians = np.median(X, axis=0)
    div_j = np.sum(np.abs(X - medians), axis=0) / N
    return np.sum(div_j) / D

def get_hussain_expl_exploit(div_history):
    div_history = np.asarray(div_history)
    max_div = np.max(div_history) if np.max(div_history) > 0 else 1e-12
    expl = (div_history / max_div) * 100.0
    exploit = (np.abs(div_history - max_div) / max_div) * 100.0
    return expl, exploit

def normalized_diversity(X, lb, ub, D0=None, eps=1e-12):
    X = np.asarray(X)
    N, d = X.shape
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()
    if lb.size == 1: lb = np.full(d, lb.item())
    if ub.size == 1: ub = np.full(d, ub.item())
    range_val = ub - lb
    range_val[range_val == 0] = eps 
    Y = (X - lb) / range_val
    diff = Y[:, None, :] - Y[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2) / np.sqrt(d)
    iu, ju = np.triu_indices(N, 1)
    D_star = dist_matrix[iu, ju].mean() if iu.size > 0 else 0.0
    if D0 is None:
        return D_star
    return D_star / (D0 + eps)

def get_exploration_exploitation(X, lb, ub, D0):
    D_rel = normalized_diversity(X, lb, ub, D0=D0)
    expl = D_rel * 100.0
    exploit = (1.0 - D_rel) * 100.0
    return np.nan_to_num(expl, nan=0.0), np.nan_to_num(exploit, nan=100.0)

def summarize_run(record, conv_threshold_rel=0.1):
    div = np.asarray(record["diversity_hist"])
    D0 = record["D0"]
    safe_D0 = D0 if D0 > 1e-15 else (np.max(div) if np.max(div) > 1e-15 else 1.0)
    
    # Normalisasi untuk perhitungan konvergensi
    D_rel_star = div / (safe_D0 + 1e-12)
    expl_star = np.nan_to_num(record["exploration_hist"], nan=0.0)
    exploit_star = np.nan_to_num(record["exploitation_hist"], nan=100.0)
    
    # Metrik Hussain (diambil dari diversity yang sama)
    expl_h, exploit_h = get_hussain_expl_exploit(div)
    max_h = np.max(div) if np.max(div) > 1e-15 else 1.0
    D_rel_h = div / (max_h + 1e-12)
    
    n_total = len(expl_star)
    idx_5 = max(1, int(0.05 * n_total))
    idx_10 = max(1, int(0.10 * n_total))
    
    # 1. Early Exploration (Poin krusial dari Prof. Ajaz)
    early_5_star = np.mean(expl_star[:idx_5])
    early_10_star = np.mean(expl_star[:idx_10])
    early_5_h = np.mean(expl_h[:idx_5])
    early_10_h = np.mean(expl_h[:idx_10])
    
    # 2. AUC (Area Under Curve)
    auc_expl_star = np.trapz(expl_star, dx=1) / n_total
    auc_exploit_star = np.trapz(exploit_star, dx=1) / n_total
    auc_expl_h = np.trapz(expl_h, dx=1) / n_total
    auc_exploit_h = np.trapz(exploit_h, dx=1) / n_total
    
    # 3. Convergence Iteration (Time-to-target proxy)
    conv_star = next((i for i, v in enumerate(D_rel_star) if v < conv_threshold_rel), n_total - 1)
    conv_h = next((i for i, v in enumerate(D_rel_h) if v < conv_threshold_rel), n_total - 1)
    
    # 4. Crossover Point (Transisi Explore ke Exploit)
    xover_star = next((i for i, (a, b) in enumerate(zip(expl_star, exploit_star)) if b > a), n_total)
    xover_h = next((i for i, (a, b) in enumerate(zip(expl_h, exploit_h)) if b > a), n_total)
    
    return {
        "Final_D": float(div[-1]),
        "DStar_Mean_Expl": float(expl_star.mean()),
        "DStar_Mean_Exploit": float(exploit_star.mean()),
        "DStar_AUC_Expl": float(auc_expl_star),
        "DStar_AUC_Exploit": float(auc_exploit_star),
        "DStar_Early_5": float(early_5_star),
        "DStar_Early_10": float(early_10_star),
        "DStar_Xover": int(xover_star),
        "DStar_Conv": int(conv_star),
        "Hussain_Mean_Expl": float(expl_h.mean()),
        "Hussain_Mean_Exploit": float(exploit_h.mean()),
        "Hussain_AUC_Expl": float(auc_expl_h),
        "Hussain_AUC_Exploit": float(auc_exploit_h),
        "Hussain_Early_5": float(early_5_h),
        "Hussain_Early_10": float(early_10_h),
        "Hussain_Xover": int(xover_h),
        "Hussain_Conv": int(conv_h),
        "FE": int(record["FE"])
    }

def plot_global_summary(detailed_data, output_dir, suite_name="CEC2020"):
    first_fn = list(detailed_data.keys())[0]
    optimizers = list(detailed_data[first_fn].keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, opt in enumerate(optimizers):
        all_fitness, all_diversity = [], []
        for fname in detailed_data.keys():
            profiles = detailed_data[fname][opt]
            all_fitness.append(np.mean([p["best_f_history"] for p in profiles], axis=0))
            all_diversity.append(np.mean([p["diversity_hist"] for p in profiles], axis=0))
        m_fit, s_fit = np.mean(all_fitness, axis=0), np.std(all_fitness, axis=0)
        m_div, s_div = np.mean(all_diversity, axis=0), np.std(all_diversity, axis=0)
        iters = np.arange(1, len(m_fit) + 1)
        c = colors[idx % len(colors)]
        ax1.plot(iters, m_fit, label=opt, color=c, linewidth=2)
        ax1.fill_between(iters, np.maximum(1e-18, m_fit - s_fit), m_fit + s_fit, color=c, alpha=0.15)
        ax2.plot(iters, m_div, label=opt, color=c, linewidth=2)
        ax2.fill_between(iters, m_div - s_div, m_div + s_div, color=c, alpha=0.15)
    ax1.set_title(f"Global Convergence (Avg over {suite_name})", fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Best Fitness Value")
    ax1.legend()
    ax2.set_title(f"Global Diversity Index $D^*$ (Avg over {suite_name})", fontweight='bold')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Diversity Index")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Global_Summary_{suite_name}.png"), dpi=300)
    plt.close()

def plot_global_search_dynamics(detailed_data, output_dir, suite_name="CEC2020"):
    first_fn = list(detailed_data.keys())[0]
    optimizers = list(detailed_data[first_fn].keys())
    fig, axs = plt.subplots(1, len(optimizers), figsize=(5*len(optimizers), 5), sharey=True)
    if len(optimizers) == 1: axs = [axs]
    for idx, opt in enumerate(optimizers):
        all_expl, all_exploit = [], []
        for fname in detailed_data.keys():
            profiles = detailed_data[fname][opt]
            all_expl.append(np.mean([np.nan_to_num(p["exploration_hist"]) for p in profiles], axis=0))
            all_exploit.append(np.mean([np.nan_to_num(p["exploitation_hist"], nan=100.0) for p in profiles], axis=0))
        m_expl, s_expl = np.mean(all_expl, axis=0), np.std(all_expl, axis=0)
        m_expt, s_expt = np.mean(all_exploit, axis=0), np.std(all_exploit, axis=0)
        iters = np.arange(1, len(m_expl) + 1)
        axs[idx].plot(iters, m_expl, label='Exploration', color='#1f77b4', linewidth=2)
        axs[idx].fill_between(iters, m_expl - s_expl, m_expl + s_expl, color='#1f77b4', alpha=0.15)
        axs[idx].plot(iters, m_expt, label='Exploitation', color='#d62728', linewidth=2)
        axs[idx].fill_between(iters, m_expt - s_expt, m_expt + s_expt, color='#d62728', alpha=0.15)
        axs[idx].set_title(opt, fontweight='bold')
        axs[idx].set_xlabel("Iterations")
        if idx == 0: axs[idx].set_ylabel("Percentage (%)")
        axs[idx].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Global_Dynamics_{suite_name}.png"), dpi=300)
    plt.close()

def plot_comparison(detailed_data, fname, output_dir):
    optimizers = list(detailed_data[fname].keys())
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    custom_colors = ['#1f77b4', '#2ca02c', '#d62728', '#7f7f7f', '#bcbd22']
    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))] 
    all_fitness_positive = True
    for idx, opt_name in enumerate(optimizers):
        profiles = detailed_data[fname][opt_name]
        mean_fitness = np.mean([p["best_f_history"] for p in profiles], axis=0)
        mean_div = np.mean([p["diversity_hist"] for p in profiles], axis=0)
        if np.any(mean_fitness <= 1e-10): all_fitness_positive = False
        actual_iters = np.arange(1, len(mean_fitness) + 1)
        color = custom_colors[idx % len(custom_colors)]
        style = line_styles[idx % len(line_styles)]
        axs[0].plot(actual_iters, mean_fitness, label=opt_name, color=color, linestyle=style, linewidth=2.5)
        axs[1].plot(actual_iters, mean_div, label=opt_name, color=color, linestyle=style, linewidth=2.5)
    axs[0].set_title(f"Convergence Curves - {fname}", fontweight='bold')
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Best Fitness Value")
    axs[0].grid(True, which="both", linestyle='--', alpha=0.5)
    if all_fitness_positive: axs[0].set_yscale('log')
    axs[0].legend(loc="upper right")
    axs[1].set_title(f"Diversity Index Comparison - {fname}", fontweight='bold')
    axs[1].set_ylabel("Diversity Index (D*)")
    axs[1].set_xlabel("Iterations")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Comparison_Dynamics_{fname}.png"), dpi=300)
    plt.close()

def plot_profile(profile, title=None):
    expl = np.nan_to_num(profile["exploration_hist"], nan=0.0)
    exploit = np.nan_to_num(profile["exploitation_hist"], nan=100.0)
    div = profile["diversity_hist"]
    bestf = profile["best_f_history"]
    it = np.arange(1, len(expl) + 1)
    mark_every = max(1, len(it) // 10)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(it, expl, label="Exploration %", color='#1f77b4', linewidth=2, marker='D', markevery=mark_every, markersize=6, markerfacecolor='white')
    axs[0].plot(it, exploit, label="Exploitation %", color='#d62728', linewidth=2, marker='s', markevery=mark_every, markersize=6, markerfacecolor='white')
    axs[0].set_ylim(-5, 105)
    axs[0].set_ylabel("Percentage (%)", fontweight='bold')
    axs[0].legend(loc="upper right", frameon=True)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[1].plot(it, div, label="D* (Diversity Index)", color='#2ca02c', linewidth=2, marker='o', markevery=mark_every, markersize=6, markerfacecolor='white')
    axs[1].set_ylabel("Diversity Index", fontweight='bold')
    ax2 = axs[1].twinx()
    ax2.plot(it, bestf, color="#ff7f0e", label="Best Fitness", linestyle='--', linewidth=1.5)
    ax2.set_ylabel("Best Objective Value", fontweight='bold')
    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines + lines2, labels + labels2, loc="upper right", frameon=True)
    axs[1].set_xlabel("Iterations", fontweight='bold')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    if title: fig.suptitle(title, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_exploration_fitness_scatter(df_ind, output_dir):
    plt.figure(figsize=(10, 6))
    optimizers = df_ind['Optimizer'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, opt in enumerate(optimizers):
        subset = df_ind[df_ind['Optimizer'] == opt]
        plt.scatter(subset['DStar_AUC_Expl'], np.log10(subset['Best_Fitness'] + 1e-20), 
                    label=opt, alpha=0.4, color=colors[idx % len(colors)], edgecolors='none')
    plt.xlabel("Exploration AUC (Proposed $D^*$)", fontweight='bold')
    plt.ylabel("Log10(Best Fitness)", fontweight='bold')
    plt.title("GLOBAL Correlation: Exploration vs. Fitness", fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "Global_Scatter_Correlation.png"), dpi=300)
    plt.close()
    scatter_dir = os.path.join(output_dir, "scatter_plots")
    if not os.path.exists(scatter_dir): os.makedirs(scatter_dir)
    for fname in df_ind['Function'].unique():
        plt.figure(figsize=(8, 5))
        f_df = df_ind[df_ind['Function'] == fname]
        for idx, opt in enumerate(optimizers):
            subset = f_df[f_df['Optimizer'] == opt]
            plt.scatter(subset['DStar_AUC_Expl'], np.log10(subset['Best_Fitness'] + 1e-20), 
                        label=opt, alpha=0.6, color=colors[idx % len(colors)])
        plt.xlabel("Exploration AUC", fontweight='bold')
        plt.ylabel("Log10(Fitness)", fontweight='bold')
        plt.title(f"Scatter Correlation: {fname}", fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(scatter_dir, f"Scatter_{fname}.png"), dpi=300)
        plt.close()

def generate_correlation_report(df_ind, output_dir):
    corr_list = []
    for fname in df_ind['Function'].unique():
        f_df = df_ind[df_ind['Function'] == fname]
        c_dstar = f_df['Best_Fitness'].corr(f_df['DStar_AUC_Expl'], method='spearman')
        c_hussain = f_df['Best_Fitness'].corr(f_df['Hussain_AUC_Expl'], method='spearman')
        corr_list.append({
            "Function": fname, 
            "Spearman_DStar_vs_Fit": c_dstar, 
            "Spearman_Hussain_vs_Fit": c_hussain
        })
    df_corr = pd.DataFrame(corr_list)
    df_corr.to_csv(os.path.join(output_dir, "Correlation_Report.csv"), index=False)
    return df_corr
