
import numpy as np
import networkx as nx
import time
from collections import Counter
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Core"))
sys.path.insert(0, os.path.dirname(__file__))  
import oqc_sdp
from utils import solve_sdp_and_round, load_dynamic_data


N_ROIS = 116
T_WINDOWS = 5

ALPHA_AB = 0.16
ALPHA_BA =   0.161
T_MAX_LOCAL_SEARCH = 50

PATIENT_DATA_DIR = os.environ.get("PATIENT_DATA_DIR","data/patient")
CONTROL_DATA_DIR = os.environ.get("CONTROL_DATA_DIR", "data/control")

CVXPATH  = os.environ.get("CVXPATH", "cvx")
DIR  = os.environ.get("DIR", "tmp")
MATLAB_PATH = os.environ.get("MATLAB_PATH", "matlab")


def create_contrast_graph(dynamic_data: dict, t: int, direction='A-B') -> nx.Graph:

    
    matrices_A = [subject_data[t] for subject_data in dynamic_data['A'].values() if t in subject_data]
    matrices_B = [subject_data[t] for subject_data in dynamic_data['B'].values() if t in subject_data]
    
    A_t = np.mean(matrices_A, axis=0)
    B_t = np.mean(matrices_B, axis=0)
    
    if direction == 'A-B':
        C_t = A_t - B_t
    elif direction == 'B-A':
        C_t = B_t - A_t
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    C_t = (C_t + C_t.T) / 2
    
    G_t = nx.Graph()
    G_t.add_nodes_from(range(N_ROIS))
    
    for i in range(N_ROIS):
        for j in range(i + 1, N_ROIS):
            weight = C_t[i, j]
            if abs(weight) > 1e-9:  # Include all non-zero weights
                G_t.add_edge(i, j, weight=weight)
    
    return G_t


def extract_subgraph(G: nx.Graph, alpha: float, t: int, direction: str = 'A-B') -> tuple:

    A = np.matrix(nx.to_numpy_array(G))
    w, d = oqc_sdp._make_coefficient_matrices(A)
    P = np.matrix(w - alpha * d)
    prefix = f"t{t}_{direction.replace('-', '')}"
    return solve_sdp_and_round(G, P, A, alpha, prefix)

def Asymmetric(dynamic_data: dict, t_total: int = T_WINDOWS, 
                       alpha_AB: float = ALPHA_AB, alpha_BA: float = ALPHA_BA) -> dict:
   
    
    print("\n" + "="*70)
    print("TEMPORAL CS-P1 ANALYSIS (SDP-based, Two Directions)")
    print("="*70 + "\n")
    
    results_AB = []
    results_BA = []

    elapsed_AB = 0
    elapsed_BA = 0   
    for t in range(t_total):
        print(f"[Window {t+1}/{t_total}] t={t}")

        try:
            G_AB_t = create_contrast_graph(dynamic_data, t, direction='A-B')
            t0 = time.time()
            nodes_AB, score_AB = extract_subgraph(G_AB_t, alpha_AB, t, 'A-B')
            elapsed_AB += time.time() - t0

            results_AB.append({
                't': t, 
                'nodes': nodes_AB, 
                'score': score_AB, 
                'size': len(nodes_AB)
            })
            
            print(f"  A-B: Score={score_AB:.4f}, Size={len(nodes_AB)}")
 
        except Exception as e:
            print(f"  A-B ERROR: {e}")
  
        try:
            G_BA_t = create_contrast_graph(dynamic_data, t, direction='B-A')
            t0 = time.time()
            nodes_BA, score_BA = extract_subgraph(G_BA_t, alpha_BA, t, 'B-A')
            elapsed_BA += time.time() - t0
            
            results_BA.append({
                't': t, 
                'nodes': nodes_BA, 
                'score': score_BA, 
                'size': len(nodes_BA)
            })
            print(f"  B-A: Score={score_BA:.4f}, Size={len(nodes_BA)}")
            
        except Exception as e:
            print(f"  B-A ERROR: {e}")
        
  
        print()
    print(f"A-B Total extraction time: {elapsed_AB:.2f}s")
    print(f"B-A Total extraction time: {elapsed_BA:.2f}s")  

    print("="*70)
    print("A-B DIRECTION ANALYSIS (A > B)")
    print("="*70)
    
    best_AB = max(results_AB, key=lambda x: x['score'])
    print(f"\nBest window: t={best_AB['t']}, Score={best_AB['score']:.4f}, Size={best_AB['size']}")
    print(f"Best nodes: {sorted(best_AB['nodes'])}")
    
    all_rois_AB = []
    for r in results_AB:
        all_rois_AB.extend(r['nodes'])
    
    roi_freq_AB = Counter(all_rois_AB)
    persistent_AB = {roi for roi, count in roi_freq_AB.items() if count >= 0.6 * t_total}
    
    print(f"\nTotal unique ROIs: {len(roi_freq_AB)}")
    print(f"Persistent ROIs (≥60%): {len(persistent_AB)}")
    print(f"  ROIs: {sorted(persistent_AB)}")
    
    print(f"\nTop 10 most frequent ROIs:")
    for roi, count in roi_freq_AB.most_common(10):
        print(f"  ROI {roi}: {count}/{t_total} windows ({100*count/t_total:.1f}%)")
    

    overlaps_AB = []
    for i in range(len(results_AB) - 1):
        nodes_t = results_AB[i]['nodes']
        nodes_t_plus_1 = results_AB[i + 1]['nodes']
        
        if len(nodes_t) > 0 and len(nodes_t_plus_1) > 0:
            overlap = len(nodes_t & nodes_t_plus_1)
            union = len(nodes_t | nodes_t_plus_1)
            overlaps_AB.append(overlap / union)
    
    stability_AB = np.mean(overlaps_AB) if overlaps_AB else 0
    print(f"\nTemporal stability (consecutive Jaccard): {stability_AB:.4f}")
    

    print("\n" + "="*70)
    print("B-A DIRECTION ANALYSIS (B > A)")
    print("="*70)
    
    best_BA = max(results_BA, key=lambda x: x['score'])
    print(f"\nBest window: t={best_BA['t']}, Score={best_BA['score']:.4f}, Size={best_BA['size']}")
    print(f"Best nodes: {sorted(best_BA['nodes'])}")
    
    all_rois_BA = []
    for r in results_BA:
        all_rois_BA.extend(r['nodes'])
    
    roi_freq_BA = Counter(all_rois_BA)
    persistent_BA = {roi for roi, count in roi_freq_BA.items() if count >= 0.6 * t_total}
    
    print(f"\nTotal unique ROIs: {len(roi_freq_BA)}")
    print(f"Persistent ROIs (≥60%): {len(persistent_BA)}")
    print(f"  ROIs: {sorted(persistent_BA)}")
    
    print(f"\nTop 10 most frequent ROIs:")
    for roi, count in roi_freq_BA.most_common(10):
        print(f"  ROI {roi}: {count}/{t_total} windows ({100*count/t_total:.1f}%)")

    overlaps_BA = []
    for i in range(len(results_BA) - 1):
        nodes_t = results_BA[i]['nodes']
        nodes_t_plus_1 = results_BA[i + 1]['nodes']
        
        if len(nodes_t) > 0 and len(nodes_t_plus_1) > 0:
            overlap = len(nodes_t & nodes_t_plus_1)
            union = len(nodes_t | nodes_t_plus_1)
            overlaps_BA.append(overlap / union)
    
    stability_BA = np.mean(overlaps_BA) if overlaps_BA else 0
    print(f"\nTemporal stability (consecutive Jaccard): {stability_BA:.4f}")
    
    print("="*70 + "\n")
    
    return {
        'A-B': {
            'Results': results_AB,
            'best': best_AB,
            'persistent_rois': persistent_AB,
            'roi_frequency': dict(roi_freq_AB),
            'stability': stability_AB
        },
        'B-A': {
            'Results': results_BA,
            'best': best_BA,
            'persistent_rois': persistent_BA,
            'roi_frequency': dict(roi_freq_BA),
            'stability': stability_BA
        }
    }


def Symmetric(dynamic_data: dict, t_total: int = T_WINDOWS, 
                       alpha: float = ALPHA_AB) -> dict:

    
    print("\n" + "="*70)
    print("TEMPORAL CS-P2 ANALYSIS (SDP-based, Symmetric)")
    print("="*70 + "\n")
    
    results_sym = []
    
    elapsed_sym = 0
    for t in range(t_total):
        print(f"[Window {t+1}/{t_total}] t={t}")
        
        try:
           
            matrices_A = [subject_data[t] for subject_data in dynamic_data['A'].values() if t in subject_data]
            matrices_B = [subject_data[t] for subject_data in dynamic_data['B'].values() if t in subject_data]
            
            A_t = np.mean(matrices_A, axis=0)
            B_t = np.mean(matrices_B, axis=0)
            
           
            C_t = np.abs(A_t - B_t)
            C_t = (C_t + C_t.T) / 2
            
           
            G_sym_t = nx.Graph()
            G_sym_t.add_nodes_from(range(N_ROIS))
            
            for i in range(N_ROIS):
                for j in range(i + 1, N_ROIS):
                    weight = C_t[i, j]
                    if abs(weight) > 1e-9:
                        G_sym_t.add_edge(i, j, weight=weight)
            
            t0 = time.time()
            nodes_sym, score_sym = extract_subgraph(G_sym_t, alpha, t, 'symmetric')
            elapsed_sym += time.time() - t0
            results_sym.append({
                't': t, 
                'nodes': nodes_sym, 
                'score': score_sym, 
                'size': len(nodes_sym)
            })
            print(f"  Symmetric: Score={score_sym:.4f}, Size={len(nodes_sym)}")
            
        except Exception as e:
            print(f"  Symmetric ERROR: {e}")
        
        print()
    print(f"Symmetric Total extraction time: {elapsed_sym:.2f}s")

    print("="*70)
    print("SYMMETRIC ANALYSIS")
    print("="*70)
    
    best_sym = max(results_sym, key=lambda x: x['score'])
    print(f"\nBest window: t={best_sym['t']}, Score={best_sym['score']:.4f}, Size={best_sym['size']}")
    print(f"Best nodes: {sorted(best_sym['nodes'])}")
    
    all_rois_sym = []
    for r in results_sym:
        all_rois_sym.extend(r['nodes'])
    
    roi_freq_sym = Counter(all_rois_sym)
    persistent_sym = {roi for roi, count in roi_freq_sym.items() if count >= 0.6 * t_total}
    
    print(f"\nTotal unique ROIs: {len(roi_freq_sym)}")
    print(f"Persistent ROIs (≥60%): {len(persistent_sym)}")
    print(f"  ROIs: {sorted(persistent_sym)}")
    
    print(f"\nTop 10 most frequent ROIs:")
    for roi, count in roi_freq_sym.most_common(10):
        print(f"  ROI {roi}: {count}/{t_total} windows ({100*count/t_total:.1f}%)")
    
    # Temporal stability
    overlaps_sym = []
    for i in range(len(results_sym) - 1):
        nodes_t = results_sym[i]['nodes']
        nodes_t_plus_1 = results_sym[i + 1]['nodes']
        
        if len(nodes_t) > 0 and len(nodes_t_plus_1) > 0:
            overlap = len(nodes_t & nodes_t_plus_1)
            union = len(nodes_t | nodes_t_plus_1)
            overlaps_sym.append(overlap / union)
    
    stability_sym = np.mean(overlaps_sym) if overlaps_sym else 0
    print(f"\nTemporal stability (consecutive Jaccard): {stability_sym:.4f}")
    
    print("="*70 + "\n")
    
    return {
        'symmetric': {
            'Results': results_sym,
            'best': best_sym,
            'persistent_rois': persistent_sym,
            'roi_frequency': dict(roi_freq_sym),
            'stability': stability_sym
        }
    }


if __name__ == '__main__':
    data = load_dynamic_data(PATIENT_DATA_DIR, CONTROL_DATA_DIR, T_WINDOWS, N_ROIS)
    

    Problem = 'CS-P2'   # Change to 'CS-P1' for Asymmetric or 'CS-P2' for Symmetric
    
    if Problem == 'CS-P1':
        results = Asymmetric(
            data, 
            t_total=T_WINDOWS, 
            alpha_AB=ALPHA_AB, 
            alpha_BA=ALPHA_BA
        )
        
      
        print("\n" + "="*70)
        print("COPY THESE TO classification.py:")
        print("="*70)
        
        print("\nsubgraphs_AB = [")
        for r in results['A-B']['Results']:
            print(f"    {sorted(list(r['nodes']))},")
        print("]")
        
        print("\nsubgraphs_BA = [")
        for r in results['B-A']['Results']:
            print(f"    {sorted(list(r['nodes']))},")
        print("]")
        print("="*70)
    
    elif Problem == 'CS-P2':
        results = Symmetric(
            data,
            t_total=T_WINDOWS,
            alpha=ALPHA_AB
        )
        
        print("\n" + "="*70)
        print("COPY THESE TO classification.py:")
        print("="*70)
        
        print("\nsubgraphs_symmetric = [")
        for r in results['symmetric']['Results']:
            print(f"    {sorted(list(r['nodes']))},")
        print("]")
        print("="*70)