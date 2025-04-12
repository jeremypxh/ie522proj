import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from itertools import combinations # Needed for fractional_factorial_design


# Key parameters and functions from the paper
def get_parameters_WT():
    """
    Returns a dictionary of all x_i parameters for the WT model,
    including the enforced values x7=x8=x9=x10=2.5, x16=p['x14'],
    x20=x18, x21=x19.
    """
    p = {}
    # From the table (WT)
    p['x1'] = 16.3036
    p['x2'] = 23.6605
    p['x3'] = 8.0636
    p['x4'] = 14.8590
    p['x5'] = 31.0464
    p['x6'] = -0.2266
    p['x11'] = 19.6572
    p['x12'] = 0.0136
    p['x13'] = 28.3559
    p['x14'] = 0.0139
    p['x15'] = 28.6912
    p['x17'] = 12.3515
    p['x18'] = 0.33553
    p['x19'] = 4363.6  # 4.3636e3
    p['x22'] = 13.2674
    p['x23'] = 7.2958
    p['x24'] = 7.2504
    p['x25'] = 7.5708
    # Enforced values
    p['x7'] = 2.5
    p['x8'] = 2.5
    p['x9'] = 2.5
    p['x10'] = 2.5
    p['x16'] = p['x14'] # Corrected based on WT table x16 = x14
    # x20 = x18, x21 = x19
    p['x20'] = p['x18']
    p['x21'] = p['x19']
    return p


def get_parameters_KO():
    """
    Returns a dictionary of all x_i parameters for the ST3Gal4–/– model,
    including the enforced values x7=x8=x9=x10=2.5, x16=p['x14'],
    x20=x18, x21=x19.
    """
    p = {}
    # From the table (ST3Gal4–/–)
    p['x1'] = 38.9756
    p['x2'] = 27.6129
    p['x3'] = 8.2235
    p['x4'] = 10.7454
    p['x5'] = 25.6248
    p['x6'] = -6.6274
    p['x11'] = 17.0064
    p['x12'] = 0.0130
    p['x13'] = 31.8518
    p['x14'] = 0.0125
    p['x15'] = 29.3783
    p['x17'] = 16.9279
    p['x18'] = 0.84621
    p['x19'] = 7427.0  # 7.4270e3
    p['x22'] = 13.4213
    p['x23'] = 7.1823
    p['x24'] = 7.1504
    p['x25'] = 6.4955
    # Enforced values
    p['x7'] = 2.5
    p['x8'] = 2.5
    p['x9'] = 2.5
    p['x10'] = 2.5
    p['x16'] = p['x14'] # Corrected based on KO table x16 = x14
    # x20 = x18, x21 = x19
    p['x20'] = p['x18']
    p['x21'] = p['x19']
    return p


def get_transition_rates(V, p):
    """
    Given a voltage V and a parameter dictionary p (either WT or KO),
    compute all alpha_ij and beta_ij and return them in a dict r.
    """
    # For brevity in the code
    x1 = p['x1']; x2 = p['x2']; x3 = p['x3']; x4 = p['x4']
    x5 = p['x5']; x6 = p['x6']; x7 = p['x7']; x8 = p['x8']
    x9 = p['x9']; x10 = p['x10']; x11 = p['x11']; x12 = p['x12']
    x13 = p['x13']; x14 = p['x14']; x15 = p['x15']; x16 = p['x16']
    x17 = p['x17']; x18 = p['x18']; x19 = p['x19']; x20 = p['x20']
    x21 = p['x21']; x22 = p['x22']; x23 = p['x23']; x24 = p['x24']
    x25 = p['x25']

    # Define alpha_11, beta_11, etc. exactly as in the table:
    alpha_11 = 3.802 / (0.1027 * np.exp(-(V + x1) / 17.0) + 0.20 * np.exp(-(V + x1) / 150.0))
    beta_11 = 0.1917 * np.exp(-(V + x2) / 20.3)

    alpha_12 = 3.802 / (0.1027 * np.exp(-(V + x3) / 15.0) + 0.23 * np.exp(-(V + x3) / 150.0))
    beta_12 = 0.20 * np.exp(-(V + x4) / 20.3)

    alpha_13 = 3.802 / (0.1027 * np.exp(-(V + x5) / 12.0) + 0.25 * np.exp(-(V + x5) / 150.0))
    beta_13 = 0.22 * np.exp(-(V + x6) / 20.3)

    alpha_111 = 3.802 / (0.1027 * np.exp(-(V + x7) / 17.0) + 0.20 * np.exp(-(V + x7) / 150.0))
    alpha_112 = 3.802 / (0.1027 * np.exp(-(V + x8) / 15.0) + 0.23 * np.exp(-(V + x8) / 150.0))
    beta_111 = 0.1917 * np.exp(-(V + x9) / 20.3)
    beta_112 = 0.20 * np.exp(-(V + x10) / 20.3)

    alpha_31 = 7.0e-7 * np.exp(-(V + x11) / x23)
    beta_31 = x12 + 2.0e-5 * (V + 7.0)

    alpha_32 = 7.0e-7 * np.exp(-(V + x13) / x24)
    beta_32 = x14 + 2.0e-5 * (V + 7.0)

    alpha_33 = 7.0e-7 * np.exp(-(V + x15) / x25)
    beta_33 = x16 + 2.0e-5 * (V + 7.0)

    alpha_2 = 0.188495 * np.exp(-((V + x17) / x22) + 0.393956)
    # Avoid division by zero or near-zero if beta_13 or beta_33 become very small
    if beta_13 * beta_33 == 0:
        beta_2 = 0 # Or some other appropriate handling
    else:
        beta_2 = alpha_13 * alpha_2 * alpha_33 / (beta_13 * beta_33)

    alpha_4 = x18 * alpha_2
    beta_4 = x19 * alpha_33

    alpha_5 = x20 * alpha_33
    beta_5 = x21 * alpha_33

    # Bundle all these into a dictionary
    r = {}
    r['alpha_11'] = alpha_11
    r['beta_11'] = beta_11
    r['alpha_12'] = alpha_12
    r['beta_12'] = beta_12
    r['alpha_13'] = alpha_13
    r['beta_13'] = beta_13
    r['alpha_111'] = alpha_111
    r['alpha_112'] = alpha_112
    r['beta_111'] = beta_111
    r['beta_112'] = beta_112
    r['alpha_31'] = alpha_31
    r['beta_31'] = beta_31
    r['alpha_32'] = alpha_32
    r['beta_32'] = beta_32
    r['alpha_33'] = alpha_33
    r['beta_33'] = beta_33
    r['alpha_2'] = alpha_2
    r['beta_2'] = beta_2
    r['alpha_4'] = alpha_4
    r['beta_4'] = beta_4
    r['alpha_5'] = alpha_5
    r['beta_5'] = beta_5
    return r


def build_generator_matrix(r):
    """
    Given the dict r of rate constants, build the 9×9 Q-matrix (generator).
    State ordering:
       0: IC3
       1: IC2
       2: IF
       3: I1
       4: I2
       5: C3
       6: C2
       7: C1
       8: O
    """
    A = np.zeros((9, 9), dtype=np.float64)

    # For readability, pull out each alpha/beta we need:
    a11 = r['alpha_11']; b11 = r['beta_11']
    a12 = r['alpha_12']; b12 = r['beta_12']
    a13 = r['alpha_13']; b13 = r['beta_13']
    a111 = r['alpha_111']; b111 = r['beta_111']
    a112 = r['alpha_112']; b112 = r['beta_112']
    a31 = r['alpha_31']; b31 = r['beta_31']
    a32 = r['alpha_32']; b32 = r['beta_32']
    a33 = r['alpha_33']; b33 = r['beta_33']
    a2 = r['alpha_2']; b2 = r['beta_2']
    a4 = r['alpha_4']; b4 = r['beta_4']
    a5 = r['alpha_5']; b5 = r['beta_5']

    # 0 = IC3: out -> IC2 (a111), C3 (a31); in <- IC2 (b111), C3 (b31)
    A[0, 1] = a111; A[0, 5] = a31
    A[1, 0] = b111; A[5, 0] = b31

    # 1 = IC2: out -> IC3 (b111), IF (a112), C2 (a32); in <- IC3 (a111), IF (b112), C2 (b32)
    A[1, 0] = b111; A[1, 2] = a112; A[1, 6] = a32
    A[0, 1] = a111; A[2, 1] = b112; A[6, 1] = b32
    A[0, 1] = b111; A[0, 5] = b31
    A[1, 0] = a111; A[5, 0] = a31
    # 1 = IC2
    A[1, 0] = a111; A[1, 2] = b112; A[1, 6] = b32
    A[0, 1] = b111; A[2, 1] = a112; A[6, 1] = a32
    # 2 = IF
    A[2, 1] = a112; A[2, 3] = b4; A[2, 7] = b33; A[2, 8] = a2
    A[1, 2] = b112; A[3, 2] = a4; A[7, 2] = a33 # No b2 term here based on diagram O->IF is a2, I1->O is b2
    # 3 = I1
    A[3, 2] = a4; A[3, 4] = b5
    A[2, 3] = b4; A[4, 3] = a5; A[8, 3] = b2 # Rate b2 from I1 to O
    # 4 = I2
    A[4, 3] = a5
    A[3, 4] = b5
    # 5 = C3
    A[5, 0] = a31; A[5, 6] = b11
    A[0, 5] = b31; A[6, 5] = a11
    # 6 = C2
    A[6, 5] = a11; A[6, 1] = a32; A[6, 7] = b12
    A[5, 6] = b11; A[1, 6] = b32; A[7, 6] = a12
    # 7 = C1
    A[7, 6] = a12; A[7, 8] = b13; A[7, 2] = a33
    A[6, 7] = b12; A[8, 7] = a13; A[2, 7] = b33
    # 8 = O
    A[8, 7] = a13; A[8, 3] = b2
    A[7, 8] = b13; A[2, 8] = a2

    for i in range(9):
        A[i, i] = -np.sum(A[j, i] for j in range(9) if i != j) # Sum down the column (rates *out* of state i)

    A = np.zeros((9, 9), dtype=np.float64)
    # 0 = IC3
    A[0, 0] = - (a111 + a31)
    A[0, 1] = b111
    A[0, 5] = b31
    # 1 = IC2
    A[1, 1] = - (b111 + a112 + a32)
    A[1, 0] = a111
    A[1, 2] = b112
    A[1, 6] = b32

    A[2, 2] = - (b112 + a4 + a33 + b2)
    A[2, 1] = a112
    A[2, 3] = b4
    A[2, 7] = b33
    A[2, 8] = a2
    # 3 = I1
    A[3, 3] = - (b4 + a5 + b2)
    A[3, 2] = a4
    A[3, 4] = b5
    # 4 = I2
    A[4, 4] = - b5
    A[4, 3] = a5
    # 5 = C3
    A[5, 5] = - (b31 + a11)
    A[5, 0] = a31
    A[5, 6] = b11
    # 6 = C2
    A[6, 6] = - (b11 + b32 + a12)
    A[6, 5] = a11
    A[6, 1] = a32
    A[6, 7] = b12
    # 7 = C1
    A[7, 7] = - (b12 + a13 + b33)
    A[7, 6] = a12
    A[7, 8] = b13
    A[7, 2] = a33
    # 8 = O
    A[8, 8] = - (b13 + a2)
    A[8, 7] = a13
    A[8, 3] = b2

    A = np.zeros((9, 9), dtype=np.float64)
    # Rates j -> i into A[i,j]
    A[1, 0] = a111; A[5, 0] = a31  # From 0 (IC3)
    A[0, 1] = b111; A[2, 1] = a112; A[6, 1] = a32  # From 1 (IC2)
    A[1, 2] = b112; A[3, 2] = a4;  A[7, 2] = a33  # From 2 (IF)
    A[2, 3] = b4;  A[4, 3] = a5;  A[8, 3] = b2   # From 3 (I1)
    A[3, 4] = b5                           # From 4 (I2)
    A[0, 5] = b31; A[6, 5] = a11            # From 5 (C3)
    A[5, 6] = b11; A[1, 6] = b32; A[7, 6] = a12  # From 6 (C2)
    A[6, 7] = b12; A[8, 7] = a13; A[2, 7] = b33  # From 7 (C1)
    A[7, 8] = b13; A[2, 8] = a2             # From 8 (O)

    # Diagonal elements A[i,i] = - sum(rates leaving state i)
    A[0, 0] = -(a111 + a31)  # Leaving IC3
    A[1, 1] = -(b111 + a112 + a32) # Leaving IC2
    A[2, 2] = -(b112 + a4 + a33) # Leaving IF
    A[3, 3] = -(b4 + a5 + b2)  # Leaving I1
    A[4, 4] = -(b5)           # Leaving I2
    A[5, 5] = -(b31 + a11)  # Leaving C3
    A[6, 6] = -(b11 + b32 + a12) # Leaving C2
    A[7, 7] = -(b12 + a13 + b33) # Leaving C1
    A[8, 8] = -(b13 + a2)  # Leaving O
    return A


def compute_stationary_distribution(A):
    """
    Solve A*pi = 0 with sum(pi)=1 for the column vector pi.
    (Note: A is the generator for dP/dt = A*P)
    Returns pi as a 1D numpy array of length 9.
    """
    A_mod = A.copy()
    A_mod[-1, :] = 1.0
    b = np.zeros(9)
    b[-1] = 1.0

    try:
        pi = np.linalg.solve(A_mod, b)
        # Check for negative probabilities due to numerical issues
        if np.any(pi < -1e-6): # Allow small negative due to precision
             print(f"Warning: Negative probabilities found in stationary distribution: {pi}")
             # Fallback: Eigenvector method
             eigenvalues, eigenvectors = np.linalg.eig(A)
             # Find eigenvector corresponding to eigenvalue closest to 0
             zero_eig_idx = np.argmin(np.abs(eigenvalues))
             pi = np.real(eigenvectors[:, zero_eig_idx])
             pi = pi / np.sum(pi) # Normalize
             if np.any(pi < 0):
                  print(f"Warning: Eigenvector method also resulted in negative probabilities: {pi}")
                  pi = np.maximum(pi, 0) # Force non-negative
                  pi = pi / np.sum(pi) # Re-normalize
        pi = pi / np.sum(pi) # Ensure normalization
    except np.linalg.LinAlgError:
        print("Warning: Linear solver failed. Trying eigenvector method.")
        # Alternative: Find the eigenvector corresponding to eigenvalue 0
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # Find eigenvector corresponding to eigenvalue closest to 0
        zero_eig_idx = np.argmin(np.abs(eigenvalues))
        pi = np.real(eigenvectors[:, zero_eig_idx])
        pi = pi / np.sum(pi) # Normalize
        if np.any(pi < 0):
            print(f"Warning: Eigenvector method resulted in negative probabilities: {pi}")
            pi = np.maximum(pi, 0) # Force non-negative
            pi = pi / np.sum(pi) # Re-normalize

    return pi


def get_stationary_distribution(V, param_dict):
    """
    High-level function:
      1) Compute all transition rates at voltage V using param_dict.
      2) Build the generator matrix A (for dP/dt = A*P).
      3) Solve for and return the stationary distribution (length-9 array).
    """
    rates = get_transition_rates(V, param_dict)
    A = build_generator_matrix(rates)
    pi = compute_stationary_distribution(A)
    return pi


def compute_SSA_SSI(voltages, param_dict):
    """
    Compute steady-state activation (SSA) and inactivation (SSI) curves
    for a range of voltages using the given parameter set.
    """
    ssa = []
    ssi = []

    for V in voltages:
        pi = get_stationary_distribution(V, param_dict)

        # SSA = probability of open state + (1-close state)
        pO = 1-np.sum(pi[5:7])+pi[8]
        ssa.append(pO)

        ssi_value = np.sum(pi[0:5])
        ssi.append(ssi_value) # Sum of inactivated states

    return np.array(ssa), np.array(ssi)


def plot_SSA_SSI_comparison(voltages, ssa_wt, ssi_wt, ssa_ko, ssi_ko, title_suffix=""):
    """
    Plot steady-state activation and inactivation curves for WT and KO.
    """
    plt.figure(figsize=(10, 6))

    # Plot WT curves
    plt.plot(voltages, ssa_wt, 'r-^', label='WT: SSA (Open Prob)')
    plt.plot(voltages, 1.0 - ssi_wt, 'b-^', label='WT: Availability (1-SSI)') # Plot availability

    # Plot KO curves
    plt.plot(voltages, ssa_ko, 'r--o', label='KO: SSA (Open Prob)')
    plt.plot(voltages, 1.0 - ssi_ko, 'b--o', label='KO: Availability (1-SSI)') # Plot availability

    plt.xlabel('Voltage (mV)')
    plt.ylabel('Steady-State Probability')
    plt.title(f'Steady-State Activation & Availability {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 1.05) # Standard probability range


def simulate_AP(t_max=200, dt=0.1):
    """
    Simulate a simplified action potential waveform for demonstration
    """
    t = np.arange(0, t_max + dt, dt) # Ensure t_max is included
    V = np.zeros_like(t)

    # Resting potential
    V[:] = -80

    # Upstroke (stimulus applied implicitly before t=10ms)
    upstroke_start_t = 10
    upstroke_end_t = 15
    peak_v = 20

    upstroke_start_idx = int(upstroke_start_t / dt)
    upstroke_end_idx = int(upstroke_end_t / dt)
    if upstroke_end_idx > upstroke_start_idx:
         V[upstroke_start_idx:upstroke_end_idx] = np.linspace(-80, peak_v, upstroke_end_idx - upstroke_start_idx)

    # Plateau and repolarization
    plateau_end_t = 50
    repol_end_t = 150

    plateau_end_idx = int(plateau_end_t / dt)
    repol_end_idx = int(repol_end_t / dt)

    V[upstroke_end_idx:plateau_end_idx] = peak_v
    if repol_end_idx > plateau_end_idx:
        V[plateau_end_idx:repol_end_idx] = np.linspace(peak_v, -80, repol_end_idx - plateau_end_idx)
    V[repol_end_idx:] = -80 # Ensure return to rest

    return t, V


def ode_rhs(t, P, V_func, param_dict, A_cache):
    """
    Right-hand side of the ODE: dP/dt = A(V)*P for time-varying V
    Uses caching for the matrix A if voltage doesn't change much.
    """
    V = V_func(t)  # Get voltage at time t

    rates = get_transition_rates(V, param_dict)
    A = build_generator_matrix(rates)

    return A.dot(P)


def simulate_channel_dynamics(t, V, param_dict, P0=None):
    """
    Simulate the dynamics of the Markov model given an AP waveform V(t).
    """
    # Create a function to interpolate voltage at any time t
    V_func = interp1d(t, V, kind='linear', bounds_error=False, fill_value=(V[0], V[-1]))

    # Initial state distribution
    if P0 is None:
        # Calculate stationary distribution at initial voltage V[0]
        print(f"Calculating initial state distribution at V = {V[0]} mV")
        P0 = get_stationary_distribution(V[0], param_dict)
        print(f"Initial P0: {P0}")

    # Cache for matrix A (optional, might not be effective with continuous V)
    A_cache = {'V': None, 'A': None}

    # Solve the ODE system dP/dt = A(V(t)) * P
    sol = solve_ivp(
        fun=lambda t_eval, y: ode_rhs(t_eval, y, V_func, param_dict, A_cache),
        t_span=(t[0], t[-1]),
        y0=P0,
        method='BDF',  # Suitable for stiff ODEs common in Markov models
        t_eval=t,      # Evaluate solution at the same time points as V
        rtol=1e-6,     # Relative tolerance
        atol=1e-8      # Absolute tolerance
    )

    if not sol.success:
        print(f"Warning: ODE solver failed! ({sol.message})")
        # Return something sensible, e.g., P0 replicated, or NaN
        P_sol = np.full((len(t), len(P0)), np.nan)
        return P_sol


    # The solution P_sol is in sol.y (shape is (n_states, n_times))
    P_sol = sol.y.T  # Transpose to get shape (n_times, n_states)

    # Ensure probabilities sum to 1 at each time step (due to potential numerical drift)
    P_sol = P_sol / np.sum(P_sol, axis=1, keepdims=True)

    return P_sol


def plot_state_distributions(t, V, P_sol_WT, P_sol_KO, time_points=[11, 20, 50, 100, 150]):
    """
    Plot state distributions at specific time points during the AP
    Similar to Figure 13 in the paper
    """
    state_labels = ['IC3', 'IC2', 'IF', 'I1', 'I2', 'C3', 'C2', 'C1', 'O']
    n_states = len(state_labels)
    n_time_points = len(time_points)

    fig, axs = plt.subplots(n_time_points, 3, figsize=(15, 4 * n_time_points), squeeze=False)

    for i, time_ms in enumerate(time_points):
        # Find the index closest to the desired time
        time_idx = np.argmin(np.abs(t - time_ms))
        actual_time = t[time_idx]

        # Plot AP waveform with current time point highlighted
        axs[i, 0].plot(t, V, 'k-')
        axs[i, 0].plot(actual_time, V[time_idx], 'ro', markersize=8)
        axs[i, 0].set_xlim(t[0], t[-1])
        axs[i, 0].set_ylim(np.min(V)-5, np.max(V)+5)
        axs[i, 0].set_ylabel('V (mV)')
        axs[i, 0].set_title(f'Time ≈ {actual_time:.1f} ms')
        axs[i, 0].grid(True)

        # WT state distribution
        if P_sol_WT is not None and P_sol_WT.shape[0] > time_idx:
            prob_WT = P_sol_WT[time_idx]
            axs[i, 1].bar(range(n_states), prob_WT, color='skyblue', label='WT')
            axs[i, 1].set_xticks(range(n_states))
            axs[i, 1].set_xticklabels(state_labels, rotation=45, ha='right')
            axs[i, 1].set_ylim(0, 1)
            axs[i, 1].set_title('WT State Distribution')
            axs[i, 1].set_ylabel('Probability')
            axs[i, 1].grid(axis='y')
            # Print probabilities
            for j, p in enumerate(prob_WT):
                if p > 0.02:
                    axs[i, 1].text(j, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)

        # KO state distribution
        if P_sol_KO is not None and P_sol_KO.shape[0] > time_idx:
            prob_KO = P_sol_KO[time_idx]
            axs[i, 2].bar(range(n_states), prob_KO, color='salmon', label='KO')
            axs[i, 2].set_xticks(range(n_states))
            axs[i, 2].set_xticklabels(state_labels, rotation=45, ha='right')
            axs[i, 2].set_ylim(0, 1)
            axs[i, 2].set_title('ST3Gal4-/- State Distribution')
            axs[i, 2].grid(axis='y')
            # Print probabilities
            for j, p in enumerate(prob_KO):
                if p > 0.02:
                    axs[i, 2].text(j, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def compute_current_density(V, param_dict, G_Na=12.0, E_Na=40.0):
    """
    Compute the steady-state current density for a given voltage
    I_Na = G_Na * P_O * (V - E_Na)
    """
    pi = get_stationary_distribution(V, param_dict)
    pO = pi[8]  # Probability of open state
    I_Na = G_Na * pO * (V - E_Na)
    return I_Na


def plot_current_voltage_relationship(voltages, p_WT, p_KO, G_Na=12.0, E_Na=40.0):
    """
    Plot steady-state current-voltage relationship for WT and KO
    """
    I_WT = []
    I_KO = []

    voltages_dense = np.linspace(voltages[0], voltages[-1], 100) # Use dense range for smooth plot

    for V in voltages_dense:
        I_WT.append(compute_current_density(V, p_WT, G_Na, E_Na))
        I_KO.append(compute_current_density(V, p_KO, G_Na, E_Na))

    plt.figure(figsize=(10, 6))
    plt.plot(voltages_dense, I_WT, 'r-', label='WT')
    plt.plot(voltages_dense, I_KO, 'b--', label='ST3Gal4-/-')
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Current Density (pA/pF)') # Assuming G_Na is in mS/uF or nS/pF -> I is pA/pF
    plt.title('Steady-State Current-Voltage Relationship')
    plt.legend()
    plt.grid(True)


def simulate_refractory_period_comparison(p_WT, p_KO, p_opt=None):
    """
    Simulate and compare refractory periods using a two-pulse protocol.
    Determines the minimum pulse interval allowing 50% recovery of the peak open probability.
    """
    # --- Simulation Parameters ---
    dt = 0.1  # ms, time step
    resting_potential = -80.0 # mV
    pulse1_start = 10.0 # ms
    pulse_dur = 2.0 # ms
    pulse_amp_mV = 20.0 # mV (depolarization relative to rest, could be absolute V)
    pulse_intervals = np.arange(10, 150, 10) # ms, test intervals for pulse 2
    t_max_per_interval = pulse1_start + max(pulse_intervals) + pulse_dur + 50 # ms, ensure enough time after pulse 2
    response_measure_window = 10 # ms, window after pulse start to find peak P(O)

    models = {'WT': p_WT, 'KO': p_KO}
    if p_opt:
        models['Optimized'] = p_opt

    results = {name: {'intervals': [], 'peak_PO_pulse1': None, 'peak_PO_pulse2': []} for name in models}
    recovery_threshold = 0.5 # 50% recovery

    # --- Simulate Pulse 1 Response (Reference) ---
    t_ref = np.arange(0, pulse1_start + pulse_dur + response_measure_window + 1, dt)
    V_ref = np.full_like(t_ref, resting_potential)
    p1_start_idx = int(pulse1_start / dt)
    p1_end_idx = int((pulse1_start + pulse_dur) / dt)
    V_ref[p1_start_idx:p1_end_idx] = resting_potential + pulse_amp_mV # Simple step

    print("Simulating reference pulse 1...")
    for name, params in models.items():
        P_sol_ref = simulate_channel_dynamics(t_ref, V_ref, params)
        po_ref = P_sol_ref[:, 8]
        # Find peak P(O) during/shortly after pulse 1
        measure_start_idx = p1_start_idx
        measure_end_idx = int((pulse1_start + pulse_dur + response_measure_window) / dt)
        results[name]['peak_PO_pulse1'] = np.max(po_ref[measure_start_idx:min(measure_end_idx, len(po_ref))])
        print(f"  {name} Peak P(O) Pulse 1: {results[name]['peak_PO_pulse1']:.4f}")

    # --- Simulate Two-Pulse Protocol for Different Intervals ---
    print("Simulating two-pulse protocol...")
    for interval in pulse_intervals:
        print(f"  Interval: {interval} ms")
        pulse2_start = pulse1_start + interval
        t_max_current = pulse2_start + pulse_dur + response_measure_window + 1
        t = np.arange(0, t_max_current, dt)
        V = np.full_like(t, resting_potential)

        # Apply pulses
        p1_start_idx = int(pulse1_start / dt)
        p1_end_idx = int((pulse1_start + pulse_dur) / dt)
        p2_start_idx = int(pulse2_start / dt)
        p2_end_idx = int((pulse2_start + pulse_dur) / dt)

        V[p1_start_idx:p1_end_idx] = resting_potential + pulse_amp_mV
        V[p2_start_idx:p2_end_idx] = resting_potential + pulse_amp_mV

        for name, params in models.items():
            P_sol = simulate_channel_dynamics(t, V, params)
            po = P_sol[:, 8]
            # Find peak P(O) during/shortly after pulse 2
            measure_start_idx = p2_start_idx
            measure_end_idx = int((pulse2_start + pulse_dur + response_measure_window) / dt)
            peak_po_pulse2 = np.max(po[measure_start_idx:min(measure_end_idx, len(po))])

            results[name]['intervals'].append(interval)
            results[name]['peak_PO_pulse2'].append(peak_po_pulse2)

    # --- Calculate Recovery and Refractory Period ---
    refractory_periods = {}
    recovery_data = {}

    print("\nCalculating refractory periods (50% recovery):")
    for name in models:
        peak_ref = results[name]['peak_PO_pulse1']
        recovery = np.array(results[name]['peak_PO_pulse2']) / peak_ref
        recovery_data[name] = recovery
        intervals = np.array(results[name]['intervals'])

        # Find the first interval where recovery >= threshold
        idx = np.where(recovery >= recovery_threshold)[0]
        if len(idx) > 0:
            # Interpolate for better estimate
            first_recovery_idx = idx[0]
            if first_recovery_idx == 0: # Recovered by the first interval tested
                 rp = intervals[0]
            else:
                 # Linear interpolation between point before and point at threshold
                 x1, y1 = intervals[first_recovery_idx - 1], recovery[first_recovery_idx - 1]
                 x2, y2 = intervals[first_recovery_idx], recovery[first_recovery_idx]
                 if y2 - y1 > 1e-6: # Avoid division by zero if recovery is flat
                     rp = x1 + (x2 - x1) * (recovery_threshold - y1) / (y2 - y1)
                 else:
                     rp = intervals[first_recovery_idx] # Assign the interval if flat
            refractory_periods[name] = rp
            print(f"  {name}: {rp:.2f} ms")
        else:
            refractory_periods[name] = np.inf # Did not recover within tested intervals
            print(f"  {name}: > {intervals[-1]} ms (Did not recover to 50%)")

    # --- Plot Recovery Curves ---
    plt.figure(figsize=(10, 6))
    colors = {'WT': 'r', 'KO': 'b', 'Optimized': 'g'}
    markers = {'WT': '^', 'KO': 'o', 'Optimized': 's'}
    linestyles = {'WT': '-', 'KO': '--', 'Optimized': ':'}

    for name in models:
        plt.plot(results[name]['intervals'], recovery_data[name],
                 marker=markers[name], linestyle=linestyles[name], color=colors[name],
                 label=f"{name} (RP50: {refractory_periods[name]:.2f} ms)")

    plt.axhline(recovery_threshold, color='grey', linestyle='--', label='50% Recovery Threshold')
    plt.xlabel('Pulse Interval (ms)')
    plt.ylabel('Peak P(Open) Recovery (Pulse 2 / Pulse 1)')
    plt.title('Refractory Period: Recovery of Peak Open Probability')
    plt.legend()
    plt.grid(True)
    # plt.show() # Removed

    return refractory_periods


# Statistical Metamodeling Implementation

def get_parameters_WT_levels():
    """
    Returns ranges (lower, upper) for WT parameters (±20%).
    """
    p = get_parameters_WT()
    levels = {}
    for key, val in p.items():
        if key in ['x7', 'x8', 'x9', 'x10']: # Keep fixed params fixed
            levels[key] = (val, val)
            continue
        if key in ['x16', 'x20', 'x21']: # Skip derived params
             continue

        factor = 1.2 if val == 0 else 0.2 # 20% variation
        low = val - abs(val * factor)
        high = val + abs(val * factor)
        if val < 0: # Ensure low < high for negative numbers
            low, high = high, low
        elif val == 0:
             low, high = -0.1, 0.1 # Or some other small range for zero params

        # Ensure derived parameters stay consistent within the range if base changes
        if key == 'x14':
             p['x16'] = p['x14'] # Update nominal before calculating range
             levels['x16'] = (p['x16'] * 0.8, p['x16'] * 1.2)
        if key == 'x18':
             p['x20'] = p['x18']
             levels['x20'] = (p['x20'] * 0.8, p['x20'] * 1.2)
        if key == 'x19':
             p['x21'] = p['x19']
             levels['x21'] = (p['x21'] * 0.8, p['x21'] * 1.2)

        levels[key] = (low, high)

    # Add back derived params, ensuring they vary if their base varies
    if 'x14' in levels: levels['x16'] = levels['x14']
    if 'x18' in levels: levels['x20'] = levels['x18']
    if 'x19' in levels: levels['x21'] = levels['x19']


    # Define parameter names carefully - excluding fixed and derived
    base_param_names = [k for k, v in levels.items() if k not in ['x7','x8','x9','x10','x16','x20','x21'] and v[0] != v[1]]
    # Check derived params' bases are included
    if 'x14' not in base_param_names and 'x16' in p: base_param_names.append('x14') # If x16 used, x14 must vary
    if 'x18' not in base_param_names and 'x20' in p: base_param_names.append('x18')
    if 'x19' not in base_param_names and 'x21' in p: base_param_names.append('x19')
    base_param_names = sorted(list(set(base_param_names)), key=lambda x: int(x[1:]))

    return levels, base_param_names


def compute_discrepancy(p_candidate, p_nominal, voltages):
    """
    Computes summed absolute difference in SSA & (1-SSI) between candidate and nominal parameters.
    """
    discrepancy = 0.0
    for V in voltages:
        try:
            pi_cand = get_stationary_distribution(V, p_candidate)
            pi_nom = get_stationary_distribution(V, p_nominal)
        except np.linalg.LinAlgError:
            print(f"LinAlgError at V={V}. Assigning high discrepancy.")
            return 1e6 # Assign large penalty if solver fails

        # ssa = P(O)
        ssa_cand = pi_cand[8]
        ssa_nom = pi_nom[8]
        # ssi = Sum(Inactivated States) -> Availability = 1 - ssi
        avail_cand = 1.0 - np.sum(pi_cand[0:5])
        avail_nom = 1.0 - np.sum(pi_nom[0:5])

        discrepancy += abs(ssa_cand - ssa_nom) + abs(avail_cand - avail_nom)
    return discrepancy


def fractional_factorial_design(param_names, levels, n_runs=None):
    """
    Generate a fractional factorial design for parameter screening.
    NOTE: This manual generator might create designs with complex aliasing.
    Using a library like pyDOE2 is recommended for robust designs.
    """
    k = len(param_names)

    if n_runs is None:
        # Determine minimum number of runs (power of 2 >= k+1 for Res III)
        n_runs = 2 ** int(np.ceil(np.log2(k + 1)))
    else:
        n_runs = int(n_runs)
        if not (n_runs > 0 and ((n_runs & (n_runs - 1)) == 0)): # Check if power of 2
             print(f"Warning: n_runs={n_runs} is not a power of 2. FF designs typically use powers of 2.")


    n_base_factors = int(np.log2(n_runs))
    if n_base_factors > k :
         print(f"Warning: n_runs={n_runs} implies more base factors ({n_base_factors}) than parameters ({k}). Using full factorial for first {k} params.")
         n_base_factors = k
         n_runs = 2**k


    print(f"Generating FF design: k={k} parameters, n_runs={n_runs} (base factors={n_base_factors})")


    # Create full factorial for base factors
    base_design = np.array(np.meshgrid(*([[-1, 1]] * n_base_factors))).T.reshape(-1, n_base_factors)

    # Add columns for remaining factors using interaction generators
    design_matrix = np.zeros((n_runs, k))
    design_matrix[:, :n_base_factors] = base_design

    current_col = n_base_factors
    gen_degree = 2 # Start with 2-factor interactions

    # Keep adding columns until we have k columns
    while current_col < k:
        generators = list(combinations(range(n_base_factors), gen_degree))
        for gen_indices in generators:
            if current_col >= k: break
            design_matrix[:, current_col] = np.prod(base_design[:, gen_indices], axis=1)
            current_col += 1
        gen_degree += 1
        if not generators: # Should not happen if k > n_base_factors
             print("Error: Could not generate enough factors for FF design.")
             break


    # Create list of parameter dictionaries for each run
    design_candidates = []
    design_signs = []
    p_nom = get_parameters_WT() # Base nominal parameters

    for i in range(n_runs):
        candidate = p_nom.copy() # Start with nominal
        candidate_signs = {}
        for j, param_name in enumerate(param_names):
            sign = design_matrix[i, j]
            candidate_signs[param_name] = sign
            low, high = levels[param_name]
            candidate[param_name] = low if sign == -1 else high

        # Update dependent parameters
        if 'x14' in candidate: candidate['x16'] = candidate['x14']
        if 'x18' in candidate: candidate['x20'] = candidate['x18']
        if 'x19' in candidate: candidate['x21'] = candidate['x19']

        design_candidates.append(candidate)
        design_signs.append(candidate_signs)

    return design_candidates, design_signs, design_matrix


def compute_factor_effects(param_names, sign_assign, discrepancies):
    """
    Compute main effects for each parameter from a 2-level design.
    Effect = (Avg response at High level) - (Avg response at Low level)
    """
    effects = {}
    n_runs = len(discrepancies)

    for param in param_names:
        sum_high = 0.0
        sum_low = 0.0
        count_high = 0
        count_low = 0

        for i, signs in enumerate(sign_assign):
            if signs[param] == +1:
                sum_high += discrepancies[i]
                count_high += 1
            else: # sign == -1
                sum_low += discrepancies[i]
                count_low += 1

        if count_high == 0 or count_low == 0:
             effects[param] = 0 # Should not happen in a balanced design
             print(f"Warning: No runs found for high/low level of {param}")
        else:
             avg_high = sum_high / count_high
             avg_low = sum_low / count_low
             effects[param] = avg_high - avg_low # Definition of main effect

    return effects


def plot_half_normal(effects):
    """
    Create a half-normal probability plot of parameter effects (absolute values).
    """
    abs_effects = {p: abs(e) for p, e in effects.items()}
    sorted_effects = sorted(abs_effects.items(), key=lambda item: item[1])
    params_sorted = [item[0] for item in sorted_effects]
    abs_effects_sorted = np.array([item[1] for item in sorted_effects])

    n = len(abs_effects_sorted)
    # Calculate theoretical quantiles for half-normal distribution
    prob = (np.arange(n) + 0.5) / n # Ranks to probabilities
    quantiles = norm.ppf(0.5 + prob / 2.0) # Inverse CDF of N(0,1) folded

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(quantiles, abs_effects_sorted, 'o', markersize=5)

    for i in range(n):
        ax.text(quantiles[i], abs_effects_sorted[i], f' {params_sorted[i]}', fontsize=8, va='bottom')

    ax.set_xlabel('Half-Normal Quantiles Z')
    ax.set_ylabel('Sorted |Effect|')
    ax.set_title('Half-Normal Plot of Effects')
    ax.grid(True)

    return fig


def lhs_design(n_samples, n_dims, ranges):
    """ Generate LHS design scaled to parameter ranges """
    sampler = qmc.LatinHypercube(d=n_dims, seed=42) # Use seed for reproducibility
    design_01 = sampler.random(n=n_samples)
    design = qmc.scale(design_01, [r[0] for r in ranges], [r[1] for r in ranges])
    return design


def create_gaussian_process_model(X, y):
    """ Create and fit a GP model using sklearn """
    # Define kernel: Constant * RBF
    # Length scale bounds are important; start broad, let optimizer find good values
    length_scale = [1.0] * X.shape[1] # Initial guess
    length_scale_bounds = (1e-2, 1e2) # Broad bounds
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)

    # Add WhiteKernel for noise level estimation if needed
    # from sklearn.gaussian_process.kernels import WhiteKernel
    # kernel += WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10, # More restarts help find better hyperparameters
        alpha=1e-8, # Small jitter for numerical stability
        random_state=42 # For reproducibility
    )
    gp.fit(X, y)
    print(f"GP fitted. Kernel: {gp.kernel_}")
    print(f"Log-Marginal-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")
    return gp


def expected_improvement(gp, X_eval, xi=0.01):
    """ Compute Expected Improvement acquisition function """
    y_mean, y_std = gp.predict(X_eval, return_std=True)
    y_std = np.maximum(y_std, 1e-9) # Avoid division by zero

    # Find the best observed value so far (minimum discrepancy)
    best_y = np.min(gp.y_train_)

    # Calculate EI
    imp = best_y - y_mean - xi
    Z = imp / y_std
    ei = imp * norm.cdf(Z) + y_std * norm.pdf(Z)
    ei[y_std < 1e-9] = 0.0 # Set EI to 0 where std dev is negligible

    return ei


def find_next_point(gp, param_bounds, n_restarts=20):
    """ Find the next point maximizing EI using multi-start optimization """
    n_dims = len(param_bounds)

    # Objective function: negative EI (since we minimize)
    def objective(x):
        # Reshape x for GP prediction
        ei_val = expected_improvement(gp, x.reshape(1, -1))
        return -ei_val[0]

    best_x = None
    min_neg_ei = np.inf

    # Generate random starting points within bounds
    # Use LHS for better coverage of the space
    start_points = lhs_design(n_restarts, n_dims, param_bounds)

    for x0 in start_points:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=param_bounds
        )
        if result.success and result.fun < min_neg_ei:
            min_neg_ei = result.fun
            best_x = result.x

    if best_x is None:
         print("Warning: Optimization for next point failed to converge from any start.")
         # Fallback: return a random point or the best start point?
         best_x = start_points[0] # Return the first random start as a fallback

    return best_x


def sequential_design_optimization(important_params, param_levels, p_target, voltages,
                                   n_initial=10, n_iterations=20, discrepancy_tol=0.01):
    """
    Perform sequential optimization using GP and EI.
    Targets parameters in p_target (e.g., KO parameters) starting from p_nominal (e.g., WT).
    """
    print("\n--- Starting Sequential Design Optimization ---")
    print(f"Targeting KO parameters. Varying {len(important_params)} important parameters.")
    print(f"Parameters to vary: {important_params}")
    print(f"Initial points: {n_initial}, Max iterations: {n_iterations}, Tolerance: {discrepancy_tol}")

    p_nominal = get_parameters_WT() # Starting point (e.g., WT)
    param_bounds = [param_levels[p] for p in important_params]

    # Scale bounds for GP internal use (optional but sometimes helps)
    scaler = StandardScaler()
    # Note: Scaling bounds isn't standard. We scale the *sampled points*.

    # Generate initial LHS design
    X_design = lhs_design(n_initial, len(important_params), param_bounds)
    y_discrepancy = []

    print(f"Evaluating {n_initial} initial design points...")
    for i in range(n_initial):
        p_candidate = p_nominal.copy()
        for j, param in enumerate(important_params):
            p_candidate[param] = X_design[i, j]
        # Update dependent parameters
        if 'x14' in important_params: p_candidate['x16'] = p_candidate['x14']
        if 'x18' in important_params: p_candidate['x20'] = p_candidate['x18']
        if 'x19' in important_params: p_candidate['x21'] = p_candidate['x19']

        disc = compute_discrepancy(p_candidate, p_target, voltages)
        y_discrepancy.append(disc)
        print(f"  Point {i+1}/{n_initial}, Discrepancy: {disc:.4f}")


    history = {'X': X_design.tolist(), 'y': y_discrepancy, 'iterations': []}

    # Main optimization loop
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")

        # Scale current design points (fit scaler only once if needed)
        # X_scaled = scaler.fit_transform(np.array(history['X'])) # Fit and transform
        X_current = np.array(history['X'])
        y_current = np.array(history['y'])

        # Fit GP model to current data
        gp = create_gaussian_process_model(X_current, y_current)

        # Find next point maximizing EI
        x_next = find_next_point(gp, param_bounds, n_restarts=10 * len(important_params)) # More restarts for higher dim

        # Check if the proposed point is too close to existing points
        from scipy.spatial.distance import cdist
        if np.min(cdist(X_current, x_next.reshape(1,-1))) < 1e-4:
             print("Warning: Next point is very close to an existing point. Stopping early.")
             break

        # Evaluate the new point
        p_candidate = p_nominal.copy()
        for j, param in enumerate(important_params):
            p_candidate[param] = x_next[j]
        # Update dependent parameters
        if 'x14' in important_params: p_candidate['x16'] = p_candidate['x14']
        if 'x18' in important_params: p_candidate['x20'] = p_candidate['x18']
        if 'x19' in important_params: p_candidate['x21'] = p_candidate['x19']

        disc = compute_discrepancy(p_candidate, p_target, voltages)
        print(f"  Evaluated next point. Discrepancy: {disc:.4f}")

        # Update history
        history['X'].append(x_next.tolist())
        history['y'].append(disc)
        history['iterations'].append({
             'iter': iteration + 1,
             'x_next': x_next.tolist(),
             'discrepancy': disc,
             'best_discrepancy': np.min(history['y'])
        })

        # Check convergence
        if disc < discrepancy_tol:
            print(f"\nConvergence criterion met (Discrepancy < {discrepancy_tol}). Stopping.")
            break
        if iteration == n_iterations - 1:
             print("\nMaximum iterations reached.")


    # Find the best parameters found
    best_idx = np.argmin(history['y'])
    best_x = history['X'][best_idx]
    p_optimal = p_nominal.copy()
    for j, param in enumerate(important_params):
        p_optimal[param] = best_x[j]
    # Update dependent parameters in optimal set
    if 'x14' in important_params: p_optimal['x16'] = p_optimal['x14']
    if 'x18' in important_params: p_optimal['x20'] = p_optimal['x18']
    if 'x19' in important_params: p_optimal['x21'] = p_optimal['x19']

    print(f"\nOptimization finished. Best discrepancy found: {history['y'][best_idx]:.4f}")
    return p_optimal, history


# --- Demonstration Functions ---

def demo_parameter_screening(p_nominal, param_levels, param_names_to_vary, voltages, n_runs=32):
    """ Demonstrate parameter screening """
    print("\n--- Starting Parameter Screening ---")
    print(f"Screening {len(param_names_to_vary)} parameters using {n_runs}-run Fractional Factorial Design.")

    # Generate design
    design_candidates, design_signs, _ = fractional_factorial_design(param_names_to_vary, param_levels, n_runs)

    # Compute discrepancies
    print(f"Evaluating {n_runs} design points...")
    discrepancies = []
    for i, cand in enumerate(design_candidates):
        disc = compute_discrepancy(cand, p_nominal, voltages)
        discrepancies.append(disc)
        # print(f"  Run {i+1}/{n_runs}, Discrepancy: {disc:.4f}")


    # Compute and analyze effects
    effects = compute_factor_effects(param_names_to_vary, design_signs, discrepancies)
    fig_half_norm = plot_half_normal(effects)

    # Sort effects by magnitude
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)

    # Identify important parameters (e.g., top N or based on plot)
    # Let's take top 10 for demonstration
    n_important = 10
    important_params = [param for param, _ in sorted_effects[:n_important]]

    print("\nParameter Effects (Sorted by Absolute Magnitude):")
    for param, effect in sorted_effects:
        print(f"  {param}: {effect:.6f}")

    print(f"\nIdentified Top {n_important} Important Parameters: {important_params}")
    print("--- Parameter Screening Finished ---")

    return important_params, fig_half_norm


def evaluate_optimized_model(p_optimal, p_WT, p_KO, voltages):
    """ Evaluate the optimized model by comparing SSA/SSI and plot """
    print("\n--- Evaluating Optimized Model ---")

    # Compute SSA and SSI for all three models
    ssa_wt, ssi_wt = compute_SSA_SSI(voltages, p_WT)
    ssa_ko, ssi_ko = compute_SSA_SSI(voltages, p_KO)
    ssa_opt, ssi_opt = compute_SSA_SSI(voltages, p_optimal)

    # Plot comparison: Activation
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(voltages, ssa_wt, 'r-^', label='WT', markersize=4)
    plt.plot(voltages, ssa_ko, 'b-o', label='KO (Target)', markersize=4)
    plt.plot(voltages, ssa_opt, 'g--s', label='Optimized', markersize=4)
    plt.xlabel('Voltage (mV)')
    plt.ylabel('SSA (Open Probability)')
    plt.title('Steady-State Activation')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 1.05)

    # Plot comparison: Availability (1-SSI)
    plt.subplot(1, 2, 2)
    plt.plot(voltages, 1.0 - ssi_wt, 'r-^', label='WT', markersize=4)
    plt.plot(voltages, 1.0 - ssi_ko, 'b-o', label='KO (Target)', markersize=4)
    plt.plot(voltages, 1.0 - ssi_opt, 'g--s', label='Optimized', markersize=4)
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Availability (1 - SSI)')
    plt.title('Steady-State Availability')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    # plt.show() # Removed

    print("--- Evaluation Finished ---")
    return (ssa_opt, ssi_opt) # Return the computed curves


def plot_optimization_history(history):
    """ Plots the discrepancy over optimization iterations """
    plt.figure(figsize=(10, 6))
    eval_num = np.arange(1, len(history['y']) + 1)
    best_y_so_far = [np.min(history['y'][:k+1]) for k in range(len(history['y']))]

    plt.plot(eval_num, history['y'], 'o-', label='Discrepancy per evaluation', markersize=4)
    plt.plot(eval_num, best_y_so_far, 'r--', label='Best discrepancy found')
    n_initial = len(history['y']) - len(history['iterations'])
    plt.axvline(n_initial + 0.5, color='grey', linestyle=':', label=f'End of Initial Design ({n_initial})')

    plt.xlabel('Evaluation Number')
    plt.ylabel('Discrepancy (vs Target)')
    plt.title('Optimization History')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Often helpful to see improvement on log scale
    # plt.show() # Removed


# Main demonstration function
def run_full_demonstration():
    """
    Run a full demonstration of the modeling, screening, and optimization.
    """
    print("=== Starting Full Demonstration ===")

    # --- Setup ---
    p_WT = get_parameters_WT()
    p_KO = get_parameters_KO() # This is the target for optimization
    voltages_coarse = np.arange(-140, 20 + 1, 10) # For optimization/screening
    voltages_fine = np.arange(-140, 20 + 1, 2)  # For final plots

    # --- Step 1: Basic Model Behavior Comparison ---
    print("\n--- Step 1: Comparing Basic WT vs KO Model Behavior ---")
    ssa_wt, ssi_wt = compute_SSA_SSI(voltages_fine, p_WT)
    ssa_ko, ssi_ko = compute_SSA_SSI(voltages_fine, p_KO)
    plot_SSA_SSI_comparison(voltages_fine, ssa_wt, ssi_wt, ssa_ko, ssi_ko, title_suffix="(WT vs KO)")
    plt.show() # Show this comparison plot

    # Simulate dynamics during AP
    t_ap, V_ap = simulate_AP(t_max=200, dt=0.1)
    print("Simulating channel dynamics during AP for WT...")
    P_sol_WT = simulate_channel_dynamics(t_ap, V_ap, p_WT)
    print("Simulating channel dynamics during AP for KO...")
    P_sol_KO = simulate_channel_dynamics(t_ap, V_ap, p_KO)
    plot_state_distributions(t_ap, V_ap, P_sol_WT, P_sol_KO, time_points=[11, 20, 50, 100, 150])
    plt.suptitle("State Distributions During Action Potential (WT vs KO)")
    plt.show() # Show state distribution plot

    # --- Step 2: Parameter Screening ---
    print("\n--- Step 2: Parameter Screening (WT parameter space) ---")
    param_levels, param_names_to_vary = get_parameters_WT_levels()
    # Screen based on discrepancy from WT nominal values
    important_params, fig_half_norm = demo_parameter_screening(p_WT, param_levels, param_names_to_vary, voltages_coarse, n_runs=32)
    plt.show() # Show half-normal plot

    # --- Step 3: Sequential Optimization ---
    print("\n--- Step 3: Sequential Optimization (Find params matching KO) ---")
    # Use WT levels as bounds, but optimize to match p_KO
    # Use only important params identified in screening
    important_param_levels = {p: param_levels[p] for p in important_params}
    p_optimal, history = sequential_design_optimization(
        important_params,
        important_param_levels,
        p_KO, # Target = KO
        voltages_coarse,
        n_initial=2 * len(important_params), # Rule of thumb: 10*dim, but keep it smaller here
        n_iterations=3 * len(important_params), # Adjust iterations based on dim
        discrepancy_tol=0.1 # Looser tolerance for demo
    )
    plot_optimization_history(history)
    plt.show() # Show optimization progress

    # --- Step 4: Evaluating Optimized Model ---
    print("\n--- Step 4: Evaluating Optimized Model Performance ---")
    _ = evaluate_optimized_model(p_optimal, p_WT, p_KO, voltages_fine)
    plt.show() # Show final SSA/SSI comparison including optimized

    # --- Step 5: Simulating Refractory Periods ---
    print("\n--- Step 5: Comparing Refractory Periods ---")
    refractory_periods = simulate_refractory_period_comparison(p_WT, p_KO, p_optimal)
    print(f"Calculated Refractory Periods (50% recovery): {refractory_periods}")
    plt.show() # Show refractory period plot

    print("\n=== Demonstration Complete! ===")

    # Optional: Save results
    results = {
        'WT_params': p_WT,
        'KO_params': p_KO,
        'optimal_params': p_optimal,
        'important_params': important_params,
        'optimization_history': history,
        'refractory_periods': refractory_periods
    }
    # np.savez('optimization_results.npz', **results) # Example save

    return results


# --- Main Execution ---
if __name__ == '__main__':

    # Option 1: Run the full demonstration including optimization
    full_results = run_full_demonstration()

    # # Option 2: Run only the basic WT vs KO comparison (comment out Option 1)
    # print("--- Running Basic WT vs KO Comparison Only ---")
    # p_WT = get_parameters_WT()
    # p_KO = get_parameters_KO()
    # voltages = np.arange(-140, 20 + 1, 2) # Fine range for plotting
    #
    # # # SSA/SSI Comparison
    # ssa_wt, ssi_wt = compute_SSA_SSI(voltages, p_WT)
    # ssa_ko, ssi_ko = compute_SSA_SSI(voltages, p_KO)
    # plot_SSA_SSI_comparison(voltages, ssa_wt, ssi_wt, ssa_ko, ssi_ko, title_suffix="(WT vs KO)")
    # plt.show()
    #
    # # # AP Dynamics
    # t_ap, V_ap = simulate_AP(t_max=200, dt=0.1)
    # print("Simulating channel dynamics during AP for WT...")
    # P_sol_WT = simulate_channel_dynamics(t_ap, V_ap, p_WT)
    # print("Simulating channel dynamics during AP for KO...")
    # P_sol_KO = simulate_channel_dynamics(t_ap, V_ap, p_KO)
    # plot_state_distributions(t_ap, V_ap, P_sol_WT, P_sol_KO, time_points=[11, 20, 50, 100, 150])
    # plt.suptitle("State Distributions During Action Potential (WT vs KO)")
    # plt.show()
    #
    # # # I-V Relationship
    # plot_current_voltage_relationship(voltages, p_WT, p_KO)
    # plt.show()
    #
    # # # Refractory Period
    # refractory_periods = simulate_refractory_period_comparison(p_WT, p_KO)
    # print(f"Calculated Refractory Periods (50% recovery): {refractory_periods}")
    # plt.show()
    # print("--- Basic Comparison Finished ---")