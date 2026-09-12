import numpy as np
import networkx as nx
import time
from typing import Tuple, Dict, List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Core"))
sys.path.insert(0, os.path.dirname(__file__)) 
import oqc_sdp
from utils import solve_sdp_and_round, load_dynamic_data


ALPHA_AB = 0.077
ALPHA_BA = 0.033
LAMBDA_VALUES = [0.3]
T_MAX_LOCAL_SEARCH = 50


N_ROIS = 116
T_WINDOWS = 12



PATIENT_DATA_DIR = os.environ.get("PATIENT_DATA_DIR", "data/patient")
CONTROL_DATA_DIR = os.environ.get("CONTROL_DATA_DIR", "data/control")

CVXPATH = os.environ.get("CVXPATH", "cvx")
DIR = os.environ.get("DIR", "tmp")




def extract_subjects_as_arrays(dynamic_data: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  
    subjects_A = []
    subjects_B = []
    
    for subj_data in dynamic_data['A'].values():
        subj_array = np.array([subj_data[t] for t in range(T_WINDOWS)])
        subjects_A.append(subj_array)
    
    for subj_data in dynamic_data['B'].values():
        subj_array = np.array([subj_data[t] for t in range(T_WINDOWS)])
        subjects_B.append(subj_array)
    
    return subjects_A, subjects_B


def group_graph(subjects: List[np.ndarray]) -> np.ndarray:

    aggregated = []
    for subject in subjects:
        avg_connectivity = np.mean(subject, axis=0)
        aggregated.append(avg_connectivity)
    return np.array(aggregated)


def summary_graph(subjects_aggregated: np.ndarray) -> np.ndarray:
   
    return np.mean(subjects_aggregated, axis=0)


def compute_edge_statistics(subjects_A: List[np.ndarray], 
                            subjects_B: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
   
    contrasts = []
    for t in range(T_WINDOWS):
        A_t = np.array([subj[t] for subj in subjects_A])
        B_t = np.array([subj[t] for subj in subjects_B])
        
        A_t_mean = np.mean(A_t, axis=0)
        B_t_mean = np.mean(B_t, axis=0)
        
        C_t = A_t_mean - B_t_mean
        contrasts.append(C_t)
    
    contrasts = np.array(contrasts)
    mean_contrast = np.mean(contrasts, axis=0)
    variance_contrast = np.var(contrasts, axis=0)
    
    return mean_contrast, variance_contrast


def score(G: nx.Graph, S: set, alpha: float, lambda_param: float,
                                 mean_contrast: np.ndarray, 
                                 variance_contrast: np.ndarray) -> float:
    
    if len(S) == 0:
        return 0.0
    
    total_score = 0.0
    for u in S:
        for v in S:
            if u < v:
                mean_val = mean_contrast[u, v]
                var_val = variance_contrast[u, v]
                edge_score = mean_val - (1 - lambda_param) * var_val - alpha
                total_score += edge_score
    
    return total_score


def extract_subgraph(G: nx.Graph, alpha: float, prefix: str,
                                        mean_contrast: np.ndarray,
                                        variance_contrast: np.ndarray,
                                        lambda_param: float) -> Tuple[set, float]:
   
    A = np.matrix(nx.to_numpy_array(G))
    w, d = oqc_sdp._make_coefficient_matrices(A)


    multi_obj_term = (1 / 4.) * (-(1 - lambda_param) * variance_contrast)
    P_multi_padded = np.zeros((w.shape[0], w.shape[1]))
    P_multi_padded[:mean_contrast.shape[0], :mean_contrast.shape[1]] = multi_obj_term

    P = np.matrix(w - alpha * d + P_multi_padded)

    nodes, _ = solve_sdp_and_round(G, P, A, alpha, prefix)
   
    final_score = score(G, nodes, alpha, lambda_param,
                                               mean_contrast, variance_contrast)
    return nodes, final_score

def localSearchNegativeOQC_MultiObjective(G: nx.Graph, S_init: set, alpha: float, 
                                         lambda_param: float,
                                         mean_contrast: np.ndarray,
                                         variance_contrast: np.ndarray,
                                         max_iterations: int = 1000) -> Tuple[set, float]:

    S_best = S_init.copy() if S_init else set()
    score_best = score(G, S_best, alpha, lambda_param, 
                                             mean_contrast, variance_contrast)
    
    nodes = list(G.nodes())
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for node in nodes:
            if node not in S_best:
                S_test = S_best | {node}
                score_test = score(G, S_test, alpha, lambda_param,
                                                         mean_contrast, variance_contrast)
                
                if score_test > score_best:
                    S_best = S_test
                    score_best = score_test
                    improved = True
        
        for node in list(S_best):
            S_test = S_best - {node}
            score_test = score(G, S_test, alpha, lambda_param,
                                                     mean_contrast, variance_contrast)
            
            if score_test > score_best:
                S_best = S_test
                score_best = score_test
                improved = True
    
    return S_best, score_best


def Difference_graph(subjects_A: List[np.ndarray], 
                                    subjects_B: List[np.ndarray],
                                    alpha: float,
                                    lambda_param: float,
                                    direction: str = 'A-B') -> Dict:
    
    print(f"\n  {direction} direction: α={alpha}, λ={lambda_param}")
    
    start_time = time.time()
    
    
    A_aggregated = group_graph(subjects_A)
    B_aggregated = group_graph(subjects_B)
    
    G_A = summary_graph(A_aggregated)
    G_B = summary_graph(B_aggregated)
    C_mean = G_A - G_B
    
    G_contrast = nx.Graph()
    G_contrast.add_nodes_from(range(N_ROIS))
    
    for i in range(N_ROIS):
        for j in range(i + 1, N_ROIS):
            weight = C_mean[i, j]
            if abs(weight) > 1e-9:
                G_contrast.add_edge(i, j, weight=weight)
    
    mean_contrast, variance_contrast = compute_edge_statistics(subjects_A, subjects_B)
    
    prefix = f"multiobjective_{direction}_{lambda_param}".replace('.', '_')
    t0 = time.time()
    S_final, score_final = extract_subgraph(
        G_contrast, alpha, prefix, mean_contrast, variance_contrast, lambda_param
    )

    elapsed_extraction = time.time() - t0
    total_time = time.time() - start_time
    

    S_list = sorted(list(S_final))
    n_edges_A = sum(1 for i, u in enumerate(S_list) for v in S_list[i+1:] if G_A[u, v] > 0)
    n_edges_B = sum(1 for i, u in enumerate(S_list) for v in S_list[i+1:] if G_B[u, v] > 0)
    contrast_sum = sum(mean_contrast[u, v] for i, u in enumerate(S_list) for v in S_list[i+1:])
    
    print(f"Extraction time: {elapsed_extraction:.2f}s")
    print(f"    Score: {score_final:.4f}, Size: {len(S_final)} ROIs, Time: {total_time:.2f}s")
    
    return {
        'direction': direction,
        'alpha': alpha,
        'lambda': lambda_param,
        'optimal_nodes': S_list,
        'optimal_score': score_final,
        'optimal_size': len(S_final),
        'n_edges_A': n_edges_A,
        'n_edges_B': n_edges_B,
        'contrast_sum': contrast_sum,
        'computation_time': total_time
    }


def Asymmetric(subjects_A: List[np.ndarray], 
                        subjects_B: List[np.ndarray],
                        alpha_AB: float,
                        alpha_BA: float,
                        lambda_param: float) -> Dict:
    
    print(f"\nRunning Multi-Objective CS-P1: λ={lambda_param}")
    print(f"Alpha values: A-B={alpha_AB}, B-A={alpha_BA}")
    
    total_start = time.time()

    print("\n--- A-B Direction (A > B) ---")
    results_AB =  Difference_graph(
        subjects_A, subjects_B, alpha_AB, lambda_param, direction='AB'
    )

    print("\n--- B-A Direction (B > A) ---")
    results_BA =  Difference_graph(
        subjects_B, subjects_A, alpha_BA, lambda_param, direction='BA'
    )
    
    total_time = time.time() - total_start
    
    return {
        'A-B': results_AB,
        'B-A': results_BA,
        'total_time': total_time,
        'lambda': lambda_param,
        'n_subjects_A': len(subjects_A),
        'n_subjects_B': len(subjects_B)
    }


def Symmetric(subjects_A: List[np.ndarray], 
                        subjects_B: List[np.ndarray],
                        alpha: float,
                        lambda_param: float) -> Dict:
    
    print(f"\nRunning Multi-Objective CS-P2: λ={lambda_param}, α={alpha}")
    
    total_start = time.time()

    A_aggregated = group_graph(subjects_A)
    B_aggregated = group_graph(subjects_B)

    G_A = summary_graph(A_aggregated)
    G_B = summary_graph(B_aggregated)
    
    C_mean = np.abs(G_A - G_B)
    
    G_contrast = nx.Graph()
    G_contrast.add_nodes_from(range(N_ROIS))
    
    for i in range(N_ROIS):
        for j in range(i + 1, N_ROIS):
            weight = C_mean[i, j]
            if abs(weight) > 1e-9:
                G_contrast.add_edge(i, j, weight=weight)
    
    contrasts = []
    for t in range(T_WINDOWS):
        A_t = np.array([subj[t] for subj in subjects_A])
        B_t = np.array([subj[t] for subj in subjects_B])
        
        A_t_mean = np.mean(A_t, axis=0)
        B_t_mean = np.mean(B_t, axis=0)
        
        C_t = np.abs(A_t_mean - B_t_mean)  
        contrasts.append(C_t)
    
    contrasts = np.array(contrasts)
    mean_contrast = np.mean(contrasts, axis=0)
    variance_contrast = np.var(contrasts, axis=0)
    
    prefix = f"multiobjective_symmetric_{lambda_param}".replace('.', '_')
    t0 = time.time()
    S_final, score_final = extract_subgraph(
        G_contrast, alpha, prefix, mean_contrast, variance_contrast, lambda_param
    )
    
    elapsed = time.time() - t0
    total_time = time.time() - total_start
    
    S_list = sorted(list(S_final))
    n_edges_A = sum(1 for i, u in enumerate(S_list) for v in S_list[i+1:] if G_A[u, v] > 0)
    n_edges_B = sum(1 for i, u in enumerate(S_list) for v in S_list[i+1:] if G_B[u, v] > 0)
    contrast_sum = sum(mean_contrast[u, v] for i, u in enumerate(S_list) for v in S_list[i+1:])
    
    print(f"Extraction time: {elapsed:.2f}s")
    print(f"  Score: {score_final:.4f}, Size: {len(S_final)} ROIs, Time: {total_time:.2f}s")
    
    return {
        'Problem': 'CS-P2',
        'alpha': alpha,
        'lambda': lambda_param,
        'optimal_nodes': S_list,
        'optimal_score': score_final,
        'optimal_size': len(S_final),
        'n_edges_A': n_edges_A,
        'n_edges_B': n_edges_B,
        'contrast_sum': contrast_sum,
        'computation_time': total_time,
        'n_subjects_A': len(subjects_A),
        'n_subjects_B': len(subjects_B)
    }


def main():
    print("="*70)
    print("Multi-Objective Contrast Subgraph Extraction")
    print("="*70)
    print(f"ROIs: {N_ROIS}, Time windows: {T_WINDOWS}")
    

    dynamic_data = load_dynamic_data(PATIENT_DATA_DIR, CONTROL_DATA_DIR, T_WINDOWS, N_ROIS)
    subjects_A, subjects_B = extract_subjects_as_arrays(dynamic_data)
    
    print(f"Loaded {len(subjects_A)} A subjects, {len(subjects_B)} B subjects\n")
    
    Problem = 'CS-P2'  # Change to 'CS-P1' for Asymmetric or 'CS-P2' for Symmetric
    
    for lambda_param in LAMBDA_VALUES:
        if Problem == 'CS-P1':
            results = Asymmetric(
                subjects_A, subjects_B, 
                alpha_AB=ALPHA_AB,
                alpha_BA=ALPHA_BA,
                lambda_param=lambda_param
            )
            

            print("\n" + "="*70)
            print("MULTI-OBJECTIVE CS-P1 RESULTS")
            print("="*70)
            print(f"Lambda: {lambda_param}")
            print(f"\nA-B: Size={results['A-B']['optimal_size']}, Nodes={results['A-B']['optimal_nodes']}")
            print(f"     Score={results['A-B']['optimal_score']:.4f}")
            print(f"\nB-A: Size={results['B-A']['optimal_size']}, Nodes={results['B-A']['optimal_nodes']}")
            print(f"     Score={results['B-A']['optimal_score']:.4f}")
            print(f"\nTotal Time: {results['total_time']:.2f}s")
            
        elif Problem == 'CS-P2':
            results = Symmetric(
                subjects_A, subjects_B,
                alpha=ALPHA_AB,
                lambda_param=lambda_param
            )
            
            print("\n" + "="*70)
            print("MULTI-OBJECTIVE CS-P2 RESULTS")
            print("="*70)
            print(f"Lambda: {lambda_param}, Alpha: {ALPHA_AB}")
            print(f"\nSymmetric: Size={results['optimal_size']}, Nodes={results['optimal_nodes']}")
            print(f"Score: {results['optimal_score']:.4f}")
            print(f"\nTotal Time: {results['computation_time']:.2f}s")
            print("="*70 + "\n")


if __name__ == "__main__":
    main()