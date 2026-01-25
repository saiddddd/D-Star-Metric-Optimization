import numpy as np
from evaluator import normalized_diversity, get_exploration_exploitation

#gwo part
def GWO_record(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    lb, ub = np.array(lb).flatten(), np.array(ub).flatten()
    if lb.size == 1: lb = np.repeat(lb, dim)
    if ub.size == 1: ub = np.repeat(ub, dim)
    Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    Alpha_pos, Alpha_score = np.zeros(dim), float("inf")
    Beta_pos, Beta_score = np.zeros(dim), float("inf")
    Delta_pos, Delta_score = np.zeros(dim), float("inf")
    
    D0 = normalized_diversity(Positions, lb, ub)
    if D0 < 1e-16: D0 = 1.0
    
    history = {"expl": [], "exploit": [], "div": [], "best_f": []}
    FE = 0
    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = objf(Positions[i, :])
            FE += 1
            if fitness < Alpha_score:
                Alpha_score, Alpha_pos = fitness, Positions[i, :].copy()
            elif fitness < Beta_score:
                Beta_score, Beta_pos = fitness, Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score, Delta_pos = fitness, Positions[i, :].copy()
        
        e, x = get_exploration_exploitation(Positions, lb, ub, D0)
        div = normalized_diversity(Positions, lb, ub)
        history["expl"].append(e); history["exploit"].append(x)
        history["div"].append(div)
        history["best_f"].append(Alpha_score)
        
        a = 2 - l * (2 / Max_iter)
        for i in range(SearchAgents_no):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2*a*r1-a, 2*r2
                X1 = Alpha_pos[j] - A1 * abs(C1*Alpha_pos[j] - Positions[i,j])
                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2*a*r1-a, 2*r2
                X2 = Beta_pos[j] - A2 * abs(C2*Beta_pos[j] - Positions[i,j])
                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2*a*r1-a, 2*r2
                X3 = Delta_pos[j] - A3 * abs(C3*Delta_pos[j] - Positions[i,j])
                Positions[i, j] = (X1 + X2 + X3) / 3.0
                
    return {"best_f": Alpha_score, "exploration_hist": np.array(history["expl"]), 
            "exploitation_hist": np.array(history["exploit"]), "diversity_hist": np.array(history["div"]),
            "best_f_history": np.array(history["best_f"]), "FE": FE, "D0": D0}

#pso part
def PSO_record(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    lb, ub = np.array(lb).flatten(), np.array(ub).flatten()
    if lb.size == 1: lb = np.repeat(lb, dim)
    if ub.size == 1: ub = np.repeat(ub, dim)
    X = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    V = np.zeros((SearchAgents_no, dim))
    PBest_X, PBest_F = X.copy(), np.full(SearchAgents_no, float("inf"))
    GBest_X, GBest_F = np.zeros(dim), float("inf")
    
    D0 = normalized_diversity(X, lb, ub)
    if D0 < 1e-16: D0 = 1.0
    
    history = {"expl": [], "exploit": [], "div": [], "best_f": []}
    FE = 0
    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            f = objf(X[i, :]); FE += 1
            if f < PBest_F[i]: PBest_F[i], PBest_X[i, :] = f, X[i, :].copy()
            if f < GBest_F: GBest_F, GBest_X = f, X[i, :].copy()
            
        e, x = get_exploration_exploitation(X, lb, ub, D0)
        div = normalized_diversity(X, lb, ub)
        history["expl"].append(e); history["exploit"].append(x)
        history["div"].append(div)
        history["best_f"].append(GBest_F)
        
        w, c1, c2 = 0.5, 1.5, 1.5
        V = w*V + c1*np.random.rand()*(PBest_X - X) + c2*np.random.rand()*(GBest_X - X)
        X = np.clip(X + V, lb, ub)
        
    return {"best_f": GBest_F, "exploration_hist": np.array(history["expl"]), 
            "exploitation_hist": np.array(history["exploit"]), "diversity_hist": np.array(history["div"]),
            "best_f_history": np.array(history["best_f"]), "FE": FE, "D0": D0}

#de part
def DE_record(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    lb, ub = np.array(lb).flatten(), np.array(ub).flatten()
    if lb.size == 1: lb = np.repeat(lb, dim)
    if ub.size == 1: ub = np.repeat(ub, dim)
    X = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    fitness = np.array([objf(ind) for ind in X], dtype=np.float64)
    FE = SearchAgents_no
    best_idx = np.argmin(fitness)
    GBest_F, GBest_X = fitness[best_idx], X[best_idx].copy()
    
    D0 = normalized_diversity(X, lb, ub)
    if D0 < 1e-16: D0 = 1.0
    
    history = {"expl": [], "exploit": [], "div": [], "best_f": []}
    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            idxs = [idx for idx in range(SearchAgents_no) if idx != i]
            abc = X[np.random.choice(idxs, 3, replace=False)]
            a, b, c = abc[0], abc[1], abc[2]
            mutant = np.clip(a + 0.5 * (b - c), lb, ub)
            cross_points = np.random.rand(dim) < 0.9
            if not np.any(cross_points): cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, X[i])
            f_trial = objf(trial); FE += 1
            if f_trial < fitness[i]:
                fitness[i], X[i] = f_trial, trial
                if f_trial < GBest_F: GBest_F, GBest_X = f_trial, trial.copy()
        
        e, x = get_exploration_exploitation(X, lb, ub, D0)
        div = normalized_diversity(X, lb, ub)
        history["expl"].append(e); history["exploit"].append(x)
        history["div"].append(div)
        history["best_f"].append(GBest_F)
        
    return {"best_f": GBest_F, "exploration_hist": np.array(history["expl"]), 
            "exploitation_hist": np.array(history["exploit"]), "diversity_hist": np.array(history["div"]),
            "best_f_history": np.array(history["best_f"]), "FE": FE, "D0": D0}

#agoa part
def AGOA_record(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    lb, ub = np.array(lb).flatten(), np.array(ub).flatten()
    if lb.size == 1: lb = np.repeat(lb, dim)
    if ub.size == 1: ub = np.repeat(ub, dim)
    x_pos = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    fitness = np.array([objf(ind) for ind in x_pos], dtype=np.float64)
    gbestfitness = np.min(fitness)
    gbestX = x_pos[np.argmin(fitness)].copy()
    
    D0 = normalized_diversity(x_pos, lb, ub)
    if D0 < 1e-16: D0 = 1.0
    
    history = {"expl": [], "exploit": [], "div": [], "best_f": []}
    Er, Dc, Co, F, chaos_factor, Dr = 0.1, 0.1, 0.1, 0.7, 0.1, 0.1
    stagnation_count = 0
    replay_buffer = []
    for l in range(Max_iter):
        sorted_indices = np.argsort(fitness)
        elite_count = int(np.ceil(SearchAgents_no * 0.2))
        normal_count = int(np.ceil(SearchAgents_no * 0.6))
        elite_indices = sorted_indices[:elite_count]
        normal_indices = sorted_indices[elite_count : elite_count+normal_count]
        worst_indices = sorted_indices[elite_count+normal_count:]
        for i in range(SearchAgents_no):
            chaos_value = chaos_factor * np.random.rand() + (1 - chaos_factor) * np.sin(np.random.rand() * np.pi)
            F_adaptive = F * chaos_value
            x1 = x_pos[np.random.choice(elite_indices)]
            x2 = x_pos[np.random.choice(normal_indices)]
            x3 = x_pos[np.random.choice(normal_indices)]
            x4 = x_pos[np.random.choice(worst_indices)]
            mutant = np.clip(Er * x1 + F_adaptive * ((x2 - x3) - Dc * x4), lb, ub)
            mutant_fitness = objf(mutant)
            if mutant_fitness < gbestfitness:
                trial, trial_fitness = mutant, mutant_fitness
            else:
                CR_adaptive = Co * chaos_value
                cross_points = np.random.rand(dim) < CR_adaptive
                if not np.any(cross_points): cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, x_pos[i])
                trial_fitness = objf(trial)
            if trial_fitness < fitness[i]:
                x_pos[i], fitness[i] = trial, trial_fitness
                stagnation_count = 0
            else: stagnation_count += 1
            if trial_fitness < gbestfitness:
                gbestfitness, gbestX = trial_fitness, trial.copy()
        
        e, x_exp = get_exploration_exploitation(x_pos, lb, ub, D0)
        div = normalized_diversity(x_pos, lb, ub)
        history["expl"].append(e); history["exploit"].append(x_exp)
        history["div"].append(div)
        history["best_f"].append(gbestfitness)
        
        if replay_buffer:
            for _ in range(min(10, len(replay_buffer))):
                rb_entry = replay_buffer[np.random.randint(len(replay_buffer))]
                diff = gbestfitness - np.min(rb_entry['fitness'])
                Er = np.clip(Er + (0.1 * diff if diff > 0 else -0.1 * diff), 0, 1)
                Dc = np.clip(Dc - (0.1 * diff if diff > 0 else -0.1 * diff), 0, 1)
        replay_buffer.append({'positions': x_pos.copy(), 'fitness': fitness.copy()})
        if len(replay_buffer) > 100: replay_buffer.pop(0)
        if div < Dr or stagnation_count > 100:
            for k in worst_indices:
                x_pos[k] = np.random.uniform(lb, ub)
                fitness[k] = objf(x_pos[k])
            stagnation_count = 0
            
    return {"best_f": gbestfitness, "exploration_hist": np.array(history["expl"]), 
            "exploitation_hist": np.array(history["exploit"]), "diversity_hist": np.array(history["div"]),
            "best_f_history": np.array(history["best_f"]), "FE": Max_iter*SearchAgents_no, "D0": D0}

#ees part
def EES_record(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    lb, ub = np.array(lb).flatten(), np.array(ub).flatten()
    if lb.size == 1: lb = np.repeat(lb, dim)
    if ub.size == 1: ub = np.repeat(ub, dim)
    def p_obj(x): return 0.2 * (1 / (np.sqrt(2 * np.pi) * 3)) * np.exp(-(x - 25) ** 2 / (2 * 3 ** 2))
    X = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    fitness_f = np.array([objf(ind) for ind in X], dtype=np.float64)
    best_idx = np.argmin(fitness_f)
    Best_fitness, best_position = fitness_f[best_idx], X[best_idx].copy()
    global_fitness, global_position = Best_fitness, best_position.copy()
    
    D0 = normalized_diversity(X, lb, ub)
    if D0 < 1e-16: D0 = 1.0
    
    history = {"expl": [], "exploit": [], "div": [], "best_f": []}
    FE = SearchAgents_no
    for t in range(1, Max_iter):
        Xnew = X.copy()
        History_pos = X.copy()
        C_param = 2 - (t / Max_iter)
        temp = np.random.rand() * 15 + 20
        xf = (best_position + global_position) / 2
        Xfood = best_position.copy()
        for i in range(SearchAgents_no):
            if temp > 30:
                if np.random.rand() < 0.5: Xnew[i, :] = X[i, :] + C_param * np.random.rand(dim) * (xf - X[i, :])
                else:
                    z = np.random.randint(0, SearchAgents_no)
                    Xnew[i, :] = X[i, :] - X[z, :] + xf
            else:
                P_val = 3 * np.random.rand() * fitness_f[i] / (objf(Xfood) + 1e-100)
                if P_val > 2:
                    Xfood_eff = np.exp(-1 / (P_val + 1e-12)) * Xfood
                    angle = 2 * np.pi * np.random.rand()
                    Xnew[i, :] = X[i, :] + (np.cos(angle) - np.sin(angle)) * Xfood_eff * p_obj(temp)
                else: Xnew[i, :] = (X[i, :] - Xfood) * p_obj(temp) + p_obj(temp) * np.random.rand(dim) * X[i, :]
        Xnew = np.clip(Xnew, lb, ub)
        for i in range(SearchAgents_no):
            new_fitness = objf(Xnew[i, :]); FE += 1
            if new_fitness < fitness_f[i]:
                fitness_f[i], X[i, :] = new_fitness, Xnew[i, :].copy()
                if new_fitness < Best_fitness: Best_fitness, best_position = new_fitness, X[i, :].copy()
            if new_fitness < global_fitness: global_fitness, global_position = new_fitness, Xnew[i, :].copy()
        
        Xff = X.copy()
        C_ees = 0.1
        for i in range(SearchAgents_no):
            if t <= 0.5 * Max_iter:
                h1 = History_pos[np.random.randint(SearchAgents_no)] if np.random.rand() < 0.85 else X[i]
                h2 = History_pos[np.random.randint(SearchAgents_no)] if np.random.rand() < 0.85 else X[i]
                Xff[i, :] = X[i, :] + (h1 - h2) * C_ees
            elif t < 0.8 * Max_iter:
                h1 = History_pos[np.random.randint(SearchAgents_no)] if np.random.rand() < 0.5 else X[i]
                h2 = History_pos[np.random.randint(SearchAgents_no)] if np.random.rand() < 0.5 else X[i]
                h3 = History_pos[np.random.randint(SearchAgents_no)] if np.random.rand() < 0.5 else X[i]
                r_vec = np.random.rand(dim)
                Xff[i, :] = X[i, :] + (r_vec * (h1 - h2) + (1 - r_vec) * (h1 - h3)) * C_ees
            else:
                h1 = History_pos[np.random.randint(SearchAgents_no)] if np.random.rand() < 0.5 else X[i]
                h2 = History_pos[np.random.randint(SearchAgents_no)] if np.random.rand() < 0.5 else X[i]
                h3 = History_pos[np.random.randint(SearchAgents_no)] if np.random.rand() < 0.5 else X[i]
                r_vec = np.random.rand(dim)
                Xff[i, :] = (X[i, :] + h3) / 2 + C_ees * r_vec * (h1 - h2)
            Xff[i, :] = np.clip(Xff[i, :], lb, ub)
            fit_ff = objf(Xff[i, :]); FE += 1
            if fit_ff < fitness_f[i]:
                fitness_f[i], X[i, :] = fit_ff, Xff[i, :].copy()
                if fit_ff < Best_fitness: Best_fitness, best_position = fit_ff, X[i, :].copy()
        
        e, x_val = get_exploration_exploitation(X, lb, ub, D0)
        div = normalized_diversity(X, lb, ub)
        history["expl"].append(e); history["exploit"].append(x_val)
        history["div"].append(div)
        history["best_f"].append(Best_fitness)
        
    return {"best_f": Best_fitness, "exploration_hist": np.array(history["expl"]), 
            "exploitation_hist": np.array(history["exploit"]), "diversity_hist": np.array(history["div"]),
            "best_f_history": np.array(history["best_f"]), "FE": FE, "D0": D0}


