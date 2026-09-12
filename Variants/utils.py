
import subprocess
import numpy as np
import scipy.io
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Core"))
sys.path.insert(0, os.path.dirname(__file__))  
import oqc_sdp
from greedy_oqc import localSearchNegativeOQC


CVXPATH = os.environ.get("CVXPATH", "cvx")
DIR = os.environ.get("DIR", "tmp")
MATLAB_PATH = os.environ.get("MATLAB_PATH", "matlab")  

T_MAX_LOCAL_SEARCH = 50

def make_matlab_script(script_path, prefix):
    with open(script_path, 'w') as f:
           f.write(f"addpath('{CVXPATH}')\n")  
           f.write("cvx_setup\n")
           f.write("load('%s.mat')\n" % prefix)
           f.write("[m,n] = size(P);\n")
           f.write("cvx_solver sedumi\n")
           f.write("cvx_begin sdp\n")
           f.write("    variable X(n, n) symmetric\n")
           f.write("    maximize trace(P*X)\n")
           f.write("    subject to\n")
           f.write("        X >= 0;\n")
           f.write("        diag(X) == ones(n, 1);\n")
           f.write("cvx_end\n")
           f.write("save('%s.txt', 'X', '-ASCII');\n" % prefix)
           f.write("quit;\n")


def solve_sdp_and_round(G, P, A, alpha, prefix):

    if not os.path.exists(DIR):
        os.makedirs(DIR, exist_ok=True)

    scipy.io.savemat(os.path.join(DIR, f"{prefix}.mat"),
                     mdict={"P": np.asarray((1 / 4.) * P)})

    script_name = f"solveSDPcontrast_{prefix}"
    script_path = os.path.join(DIR, f"{script_name}.m")
    make_matlab_script(script_path, prefix)

    script_basename = os.path.basename(script_path)
    matlab_command = (
        f"cd '{DIR}'; "
        f"'{MATLAB_PATH}' -nodisplay -nosplash < '{script_basename}'"
    )
    subprocess.call(matlab_command, shell=True)

    try:
        X = np.genfromtxt(os.path.join(DIR, f"{prefix}.txt"), dtype=float)
    except Exception:
      
        S_star, final_score = localSearchNegativeOQC(
            G, alpha, t_max=T_MAX_LOCAL_SEARCH, seed=None)
        return set(S_star.nodes()), final_score

    L = oqc_sdp.semidefinite_cholesky(X)
    nodeset, _, _ = oqc_sdp.random_projection_qp(L, P, A, alpha, t=1000)

    nodes_graph = set([i - 1 for i in nodeset if i > 0])
    S_init = G.subgraph(nodes_graph).copy() if nodes_graph else None

    S_star, final_score = localSearchNegativeOQC(
        G, alpha, t_max=T_MAX_LOCAL_SEARCH, seed=S_init)

    return set(S_star.nodes()), final_score


def load_dynamic_data(patient_dir, control_dir, t_windows, n_rois,
                      data_key="weighted_adjacency_matrices"):
    import glob
    dynamic_data = {"A": {}, "B": {}}
    print(f"Loading data...\n  A: {patient_dir}\n  B: {control_dir}")

    patient_files = glob.glob(f"{patient_dir}/*.mat")
    print(f"Found {len(patient_files)} patient files.")
    for i, file_path in enumerate(patient_files):
        try:
            data = scipy.io.loadmat(file_path)
            mats = data[data_key]
            if mats.shape == (t_windows, n_rois, n_rois):
                dynamic_data["A"][f"A_{i+1}"] = {t: mats[t, :, :] for t in range(t_windows)}
        except Exception:
            continue

    control_files = glob.glob(f"{control_dir}/*.mat")
    print(f"Found {len(control_files)} control files.")
    for i, file_path in enumerate(control_files):
        try:
            data = scipy.io.loadmat(file_path)
            mats = data[data_key]
            if mats.shape == (t_windows, n_rois, n_rois):
                dynamic_data["B"][f"B_{i+1}"] = {t: mats[t, :, :] for t in range(t_windows)}
        except Exception:
            continue

    print(f"Loaded {len(dynamic_data['A'])} A subjects and "
          f"{len(dynamic_data['B'])} B subjects.\n")
    return dynamic_data
