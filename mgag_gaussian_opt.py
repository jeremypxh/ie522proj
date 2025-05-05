

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import time
import pickle
from typing import Dict, List, Tuple, Optional, Any, Callable

from multiprocessing import Pool, cpu_count
from functools import partial


# --- Configuration & Parameters ---
SEED = 42
np.random.seed(SEED)

# Default parameters
G_NA_DEFAULT = 12.0  # Max sodium conductance
E_NA_DEFAULT = 20.5  # Sodium reversal potential (mV)
RESTING_V_DEFAULT = -80.0  # Resting potential (mV)
DEFAULT_VOLTAGES = np.arange(-120, 21, 5)  # Voltage range for analysis (mV)
NUMERICAL_EPS = 1e-10 # Small number to prevent division by zero
MAX_RATE = 1e7       # Max physiological rate to prevent numerical instability
MIN_RATE = 0.0       # Min rate

# Fixed parameters  - Values from WT Table III
FIXED_PARAMS_VALUES = {'x7': 16.3, 'x8': 8.0, 'x9': 23.6, 'x10': 14.8}

# Dependent parameters
DEPENDENT_PARAM_MAP = {'x16': 'x14', 'x20': 'x18', 'x21': 'x19'}

# State labels for the 9-state Markov model
STATE_LABELS = ['IC3', 'IC2', 'IF', 'I1', 'I2', 'C3', 'C2', 'C1', 'O']
N_STATES = len(STATE_LABELS)
OPEN_STATE_INDEX = STATE_LABELS.index('O')  # Index of the open state
INACTIVATED_STATE_INDICES = [STATE_LABELS.index(s) for s in
                             ['IC3', 'IC2', 'IF', 'I1', 'I2']]  # Indices of inactivated states

# Parameters identified as important
SSA_IMPORTANT_PARAMS = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x17', 'x22']
SSI_REC_IMPORTANT_PARAMS = ['x11', 'x12', 'x13', 'x14', 'x15', 'x18', 'x19', 'x23', 'x25']


# --- Parameter Handling ---

def _update_dependent_params(p: Dict[str, float]) -> Dict[str, float]:
    """Helper function to update dependent parameters based on their source."""
    for dep_param, src_param in DEPENDENT_PARAM_MAP.items():
        if src_param in p:
            p[dep_param] = p[src_param]
    return p

def get_parameters_WT() -> Dict[str, float]:
    """Returns a dictionary of parameters for the WT model from Table III."""
    p = {
        'x1': 16.3036, 'x2': 23.6605, 'x3': 8.0636, 'x4': 14.8590,
        'x5': 31.0464, 'x6': -0.2266,
        'x11': 19.6572, 'x12': 0.0136, 'x13': 28.3559, 'x14': 0.0139,
        'x15': 28.6912,
        'x17': 12.3515, 'x18': 0.33553, 'x19': 4363.6,  # 4.3636e3
        'x22': 13.2674, 'x23': 7.2958, 'x24': 7.2504, 'x25': 7.5708
    }
    p.update(FIXED_PARAMS_VALUES)
    p = _update_dependent_params(p)
    for i in range(1, 26):
        key = f'x{i}'
        if key not in p:
             warnings.warn(f"Parameter {key} missing from WT defaults, setting to 0.")
             p[key] = 0.0
    return p

def get_parameters_KO() -> Dict[str, float]:
    """Returns a dictionary of parameters for the ST3Gal4−/− (KO) model from Table III."""
    p = {
        'x1': 38.9756, 'x2': 27.6129, 'x3': 8.2235, 'x4': 10.7454,
        'x5': 25.6248, 'x6': -6.6274,
        'x11': 17.0064, 'x12': 0.0130, 'x13': 31.8518, 'x14': 0.0125,
        'x15': 29.3783,
        'x17': 16.9279, 'x18': 0.84621, 'x19': 7427.0,  # 7.4270e3
        'x22': 13.4213, 'x23': 7.1823, 'x24': 7.1504, 'x25': 6.4955
    }
    p.update(FIXED_PARAMS_VALUES)
    p = _update_dependent_params(p)
    for i in range(1, 26):
        key = f'x{i}'
        if key not in p:
             warnings.warn(f"Parameter {key} missing from KO defaults, setting to 0.")
             p[key] = 0.0
    return p

def get_parameter_definitions(variation: float = 0.5) -> Dict[str, Dict[str, Any]]:
    """Returns dict mapping param names to nominal values and bounds."""
    p_wt = get_parameters_WT()
    param_defs = {}
    for i in range(1, 26):
        key = f'x{i}'
        nominal_val = p_wt.get(key, 0.0)
        is_fixed = key in FIXED_PARAMS_VALUES
        is_dependent = key in DEPENDENT_PARAM_MAP
        low, high = None, None
        if not is_fixed and not is_dependent:
            if np.isclose(nominal_val, 0):
                delta = 0.5
                low, high = -delta, delta
            else:
                delta = abs(nominal_val * variation)
                low = nominal_val - delta
                high = nominal_val + delta
                if low > high: low, high = high, low
        param_defs[key] = {'nominal': nominal_val, 'low': low, 'high': high,
                           'fixed': is_fixed, 'dependent': is_dependent}
    return param_defs

# --- Core Markov Model Functions ---

def get_transition_rates(V: float, p: Dict[str, float]) -> Dict[str, float]:
    """Calculate transition rates based on voltage and parameters."""
    required_keys = [f'x{i}' for i in range(1, 26)]
    if not all(key in p for key in required_keys):
        missing = [key for key in required_keys if key not in p]
        raise ValueError(f"Missing parameters in input dictionary: {missing}")
    x1=p['x1']; x2=p['x2']; x3=p['x3']; x4=p['x4']; x5=p['x5']; x6=p['x6']
    x7=p['x7']; x8=p['x8']; x9=p['x9']; x10=p['x10']
    x11=p['x11']; x12=p['x12']; x13=p['x13']; x14=p['x14']; x15=p['x15']; x16=p['x16']
    x17=p['x17']; x18=p['x18']; x19=p['x19']; x20=p['x20']; x21=p['x21']
    x22=p['x22']; x23=p['x23']; x24=p['x24']; x25=p['x25']
    def safe_exp(val): return np.exp(np.clip(val, -700, 700))
    safe_x22 = np.maximum(abs(x22), NUMERICAL_EPS) * np.sign(x22) if not np.isclose(x22, 0) else NUMERICAL_EPS
    safe_x23 = np.maximum(abs(x23), NUMERICAL_EPS) * np.sign(x23) if not np.isclose(x23, 0) else NUMERICAL_EPS
    safe_x24 = np.maximum(abs(x24), NUMERICAL_EPS) * np.sign(x24) if not np.isclose(x24, 0) else NUMERICAL_EPS
    safe_x25 = np.maximum(abs(x25), NUMERICAL_EPS) * np.sign(x25) if not np.isclose(x25, 0) else NUMERICAL_EPS
    alpha_11_den = 0.1027 * safe_exp(-(V + x1) / 17.0) + 0.20 * safe_exp(-(V + x1) / 150.0)
    alpha_11 = 3.802 / np.maximum(alpha_11_den, NUMERICAL_EPS)
    beta_11 = 0.1917 * safe_exp(-(V + x2) / 20.3)
    alpha_12_den = 0.1027 * safe_exp(-(V + x3) / 15.0) + 0.23 * safe_exp(-(V + x3) / 150.0)
    alpha_12 = 3.802 / np.maximum(alpha_12_den, NUMERICAL_EPS)
    beta_12 = 0.20 * safe_exp(-(V + x4) / 20.3)
    alpha_13_den = 0.1027 * safe_exp(-(V + x5) / 12.0) + 0.25 * safe_exp(-(V + x5) / 15.0)
    alpha_13 = 3.802 / np.maximum(alpha_13_den, NUMERICAL_EPS)
    beta_13 = 0.22 * safe_exp(-(V + x6) / 20.3)
    alpha_111_den = 0.1027 * safe_exp(-(V + x7) / 17.0) + 0.20 * safe_exp(-(V + x7) / 150.0)
    alpha_111 = 3.802 / np.maximum(alpha_111_den, NUMERICAL_EPS)
    alpha_112_den = 0.1027 * safe_exp(-(V + x8) / 15.0) + 0.23 * safe_exp(-(V + x8) / 150.0)
    alpha_112 = 3.802 / np.maximum(alpha_112_den, NUMERICAL_EPS)
    beta_111 = 0.1917 * safe_exp(-(V + x9) / 20.3)
    beta_112 = 0.20 * safe_exp(-(V + x10) / 20.3)
    alpha_31 = 7.0e-7 * safe_exp(-(V + x11) / safe_x23)
    beta_31 = x12 + 2.0e-5 * (V + 7.0)
    alpha_32 = 7.0e-7 * safe_exp(-(V + x13) / safe_x24)
    beta_32 = x14 + 2.0e-5 * (V + 7.0)
    alpha_33 = 7.0e-7 * safe_exp(-(V + x15) / safe_x25)
    beta_33 = x16 + 2.0e-5 * (V + 7.0)
    alpha_2 = 0.188495 * safe_exp(-((V + x17) / safe_x22) + 0.393956)
    safe_beta_13 = np.maximum(beta_13, NUMERICAL_EPS)
    safe_beta_33 = np.maximum(beta_33, NUMERICAL_EPS)
    beta_2 = (alpha_13 * alpha_2 * alpha_33) / (safe_beta_13 * safe_beta_33)
    alpha_4 = x18 * alpha_2; beta_4 = x19 * alpha_33
    alpha_5 = x20 * alpha_2; beta_5 = x21 * alpha_33
    rates = {'a11': alpha_11, 'b11': beta_11, 'a12': alpha_12, 'b12': beta_12,
             'a13': alpha_13, 'b13': beta_13, 'a111': alpha_111, 'b111': beta_111,
             'a112': alpha_112, 'b112': beta_112, 'a31': alpha_31, 'b31': beta_31,
             'a32': alpha_32, 'b32': beta_32, 'a33': alpha_33, 'b33': beta_33,
             'a2': alpha_2, 'b2': beta_2, 'a4': alpha_4, 'b4': beta_4,
             'a5': alpha_5, 'b5': beta_5}
    for key in rates: rates[key] = np.clip(rates[key], MIN_RATE, MAX_RATE)
    rates['b31'] = np.maximum(rates['b31'], MIN_RATE)
    rates['b32'] = np.maximum(rates['b32'], MIN_RATE)
    rates['b33'] = np.maximum(rates['b33'], MIN_RATE)
    return rates

def build_generator_matrix(rates: Dict[str, float]) -> np.ndarray:
    """Build the generator matrix."""
    A = np.zeros((N_STATES, N_STATES))
    A[1, 0]=rates['a111']; A[5, 0]=rates['a31']; A[0, 1]=rates['b111']; A[2, 1]=rates['a112']
    A[6, 1]=rates['a32']; A[1, 2]=rates['b112']; A[3, 2]=rates['a4']; A[7, 2]=rates['b33']
    A[2, 3]=rates['b4']; A[4, 3]=rates['a5']; A[8, 3]=rates['b2']; A[3, 4]=rates['b5']
    A[0, 5]=rates['b31']; A[6, 5]=rates['a11']; A[5, 6]=rates['b11']; A[1, 6]=rates['b32']
    A[7, 6]=rates['a12']; A[6, 7]=rates['b12']; A[8, 7]=rates['a13']; A[2, 7]=rates['a33']
    A[7, 8]=rates['b13']; A[2, 8]=rates['a2']
    for i in range(N_STATES): A[i, i] = -np.sum(A[:, i])
    return A

def compute_stationary_distribution(A: np.ndarray) -> Optional[np.ndarray]:
    """Compute stationary distribution."""
    A_mod = A.T.copy(); A_mod[-1, :] = 1.0
    b = np.zeros(N_STATES); b[-1] = 1.0
    try:
        pi = np.linalg.solve(A_mod, b)
        if np.any(pi < -NUMERICAL_EPS) or abs(np.sum(pi) - 1.0) > 1e-6:
             raise np.linalg.LinAlgError("Invalid probabilities")
        pi = np.maximum(pi, 0); pi /= np.sum(pi)
        if np.linalg.norm(A @ pi) > 1e-6:
             warnings.warn("Stationary distribution solution doesn't satisfy A*pi=0 well.")
        return pi
    except np.linalg.LinAlgError:
        try:
            u, s, vh = np.linalg.svd(A.T)
            null_space_vector = vh[-1, :]; pi = np.abs(null_space_vector); pi /= np.sum(pi)
            if np.linalg.norm(A @ pi) > 1e-5:
                warnings.warn(f"SVD fallback failed A*pi=0 verification (norm={np.linalg.norm(A @ pi):.2e}).")
                return None
            if np.any(pi < -NUMERICAL_EPS) or abs(np.sum(pi) - 1.0) > 1e-6:
                 warnings.warn("SVD fallback stationary distribution invalid after normalization.")
                 return None
            pi = np.maximum(pi, 0); pi /= np.sum(pi)
            return pi
        except np.linalg.LinAlgError:
            warnings.warn("Both direct solve and SVD failed for stationary distribution.")
            return None

def get_stationary_distribution(V: float, param_dict: Dict[str, float]) -> Optional[np.ndarray]:
    """Compute stationary distribution."""
    try:
        rates = get_transition_rates(V, param_dict); A = build_generator_matrix(rates)
        return compute_stationary_distribution(A)
    except Exception as e:
        warnings.warn(f"Error computing stationary distribution at V={V}: {e}"); return None

# --- Simulation Protocols ---

def compute_SSA_SSI(voltages: np.ndarray, param_dict: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute steady-state activation (SSA) and inactivation (SSI) curves."""
    ssa = np.full_like(voltages, np.nan, dtype=float)
    ssi = np.full_like(voltages, np.nan, dtype=float)
    for i, V in enumerate(voltages):
        pi = get_stationary_distribution(V, param_dict)
        if pi is not None: ssa[i] = pi[OPEN_STATE_INDEX]; ssi[i] = np.sum(pi[INACTIVATED_STATE_INDICES])
    ssa_max = np.nanmax(ssa)
    if ssa_max is not None and ssa_max > NUMERICAL_EPS: ssa /= ssa_max
    else: ssa[:] = 0
    ssi = np.clip(ssi, 0.0, 1.0)
    return ssa, ssi

def ode_rhs(t: float, P: np.ndarray, V_func: Callable[[float], float], param_dict: Dict[str, float]) -> np.ndarray:
    """Right-hand side for the ODE solver: dP/dt = A(V(t)) * P."""
    V = V_func(t); rates = get_transition_rates(V, param_dict); A = build_generator_matrix(rates)
    return A @ P

def simulate_channel_dynamics(t_eval: np.ndarray, V_protocol: np.ndarray,
                              param_dict: Dict[str, float], P0: Optional[np.ndarray] = None
                              ) -> Optional[np.ndarray]:
    """Simulate Markov model dynamics for a given voltage protocol V(t)."""
    if P0 is None:
        P0 = get_stationary_distribution(V_protocol[0], param_dict)
        if P0 is None:
            warnings.warn(f"Failed to compute initial P0 for simulation starting at V={V_protocol[0]:.1f} mV.")
            return None
    P0 = np.maximum(P0, 0); P0 /= np.sum(P0)
    try:
        V_func = interp1d(t_eval, V_protocol, kind='previous', bounds_error=False,
                          fill_value=(V_protocol[0], V_protocol[-1]))
    except ValueError as e:
         warnings.warn(f"Could not create voltage interpolation function: {e}"); return None
    try:
        sol = solve_ivp(fun=lambda t, y: ode_rhs(t, y, V_func, param_dict),
                        t_span=(t_eval[0], t_eval[-1]), y0=P0, method='BDF',
                        t_eval=t_eval, rtol=1e-5, atol=1e-7, vectorized=False)
        if not sol.success:
            warnings.warn(f"ODE solver failed: {sol.message} at t={sol.t[-1]:.2f}ms"); return None
        P_sol = sol.y.T
        sum_P = np.sum(P_sol, axis=1, keepdims=True); sum_P[sum_P < NUMERICAL_EPS] = 1.0
        P_sol = P_sol / sum_P; P_sol = np.clip(P_sol, 0.0, 1.0)
        return P_sol
    except Exception as e:
        warnings.warn(f"Exception during ODE integration: {e}"); return None

def simulate_recovery_protocol(param_dict: Dict[str, float], dt: float = 0.1, max_interval: float = 300
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate recovery from inactivation using a two-pulse protocol."""
    hold_V = -100.0; prepulse_V = -10.0; prepulse_duration = 1000.0
    test_pulse_V = -10.0; test_pulse_duration = 50.0
    intervals = np.arange(dt, max_interval + dt, dt * 10) # ~30 intervals
    recovery_fractions = np.full_like(intervals, np.nan, dtype=float)
    P_hold = get_stationary_distribution(hold_V, param_dict)
    if P_hold is None: warnings.warn(f"Recovery: Cannot compute P_hold at {hold_V} mV."); return intervals, recovery_fractions

    # --- Reference Pulse ---
    t_ref_max = prepulse_duration + test_pulse_duration + 10.0; t_ref = np.arange(0, t_ref_max, dt)
    V_ref = np.full_like(t_ref, hold_V)
    prepulse_end_idx = int(prepulse_duration / dt); test_pulse_end_idx = int((prepulse_duration + test_pulse_duration) / dt)
    V_ref[0:prepulse_end_idx] = prepulse_V; V_ref[prepulse_end_idx:test_pulse_end_idx] = test_pulse_V
    P_sol_ref = simulate_channel_dynamics(t_ref, V_ref, param_dict, P0=P_hold)
    if P_sol_ref is None: warnings.warn("Recovery: Failed reference pulse sim."); return intervals, recovery_fractions
    peak_P_open_ref = np.max(P_sol_ref[prepulse_end_idx:test_pulse_end_idx, OPEN_STATE_INDEX])
    if peak_P_open_ref < NUMERICAL_EPS: warnings.warn("Recovery: Ref peak P(O) near zero."); peak_P_open_ref = NUMERICAL_EPS

    # --- Two-Pulse Protocol for each Interval ---
    for i, interval in enumerate(intervals):
        t_max = prepulse_duration + interval + test_pulse_duration + 10.0; t = np.arange(0, t_max, dt)
        V = np.full_like(t, hold_V)
        prepulse_end_idx_i = int(prepulse_duration / dt)
        interval_end_idx_i = int((prepulse_duration + interval) / dt)
        test_pulse_end_idx_i = int((prepulse_duration + interval + test_pulse_duration) / dt)
        V[0:prepulse_end_idx_i] = prepulse_V; V[interval_end_idx_i:test_pulse_end_idx_i] = test_pulse_V
        P_sol = simulate_channel_dynamics(t, V, param_dict, P0=P_hold)
        if P_sol is None: warnings.warn(f"Recovery: Failed sim for interval {interval:.1f} ms."); continue
        peak_P_open = np.max(P_sol[interval_end_idx_i:test_pulse_end_idx_i, OPEN_STATE_INDEX])
        recovery_fractions[i] = peak_P_open / peak_P_open_ref

    return intervals, np.clip(recovery_fractions, 0.0, 1.0)

def calculate_refractory_period(intervals: np.ndarray, recovery_fractions: np.ndarray,
                                threshold: float = 0.5) -> float:
    """Calculates refractory period by interpolating recovery curve at threshold."""
    valid_indices = np.where(~np.isnan(recovery_fractions))[0]
    if len(valid_indices) < 2: return np.nan
    valid_intervals = intervals[valid_indices]; valid_recovery = recovery_fractions[valid_indices]
    if np.nanmax(valid_recovery) < threshold: return np.nan
    if np.nanmin(valid_recovery) >= threshold: return valid_intervals[0]
    try:
        interp_func = interp1d(valid_recovery, valid_intervals, kind='linear', bounds_error=False, fill_value=np.nan)
        rp = interp_func(threshold)
        if np.isnan(rp) or rp < valid_intervals[0] or rp > valid_intervals[-1]:
             idx_above = np.where(valid_recovery >= threshold)[0]
             if not idx_above.any(): return np.nan
             first_idx = idx_above[0]
             if first_idx == 0: return valid_intervals[0]
             x1, y1 = valid_intervals[first_idx - 1], valid_recovery[first_idx - 1]
             x2, y2 = valid_intervals[first_idx], valid_recovery[first_idx]
             if abs(y2 - y1) < NUMERICAL_EPS: rp = x1
             else: rp = x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
        return rp
    except ValueError: warnings.warn("Interpolation for RP failed."); return np.nan

# --- Action Potential Simulation---

def ap_ode_rhs(t: float, y: np.ndarray, stim_func: Callable, param_dict: Dict[str, float],
               Cm: float, g_K: float, g_leak: float, E_K: float, E_leak: float, G_Na: float, E_Na: float
               ) -> np.ndarray:
    """Combined RHS for AP simulation."""
    V = y[0]; P = y[1:]
    I_Na = G_Na * P[OPEN_STATE_INDEX] * (V - E_Na)
    n_inf = 1.0 / (1.0 + np.exp(-(V + 30.0) / 10.0))
    I_K = g_K * n_inf * (V - E_K)
    I_leak = g_leak * (V - E_leak); I_stim = stim_func(t)
    dVdt = -(I_Na + I_K + I_leak - I_stim) / Cm
    try:
        rates = get_transition_rates(V, param_dict); A = build_generator_matrix(rates); dPdt = A @ P
    except Exception: dPdt = np.zeros_like(P)
    return np.concatenate(([dVdt], dPdt))

def simulate_action_potential(param_dict: Dict[str, float], duration: float = 250, dt: float = 0.1,
                              G_Na: float = G_NA_DEFAULT, E_Na: float = E_NA_DEFAULT
                              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Simulate cellular action potential."""
    resting_V = RESTING_V_DEFAULT; stim_amplitude = -20.0; stim_delay = 10.0; stim_duration = 1.0
    Cm = 1.0; g_K = 0.3; g_leak = 0.05; E_K = -90.0; E_leak = -60.0
    t_eval = np.arange(0, duration, dt)
    def stim_func(t): return stim_amplitude if stim_delay <= t < stim_delay + stim_duration else 0.0
    P_init = get_stationary_distribution(resting_V, param_dict)
    if P_init is None: warnings.warn("AP Sim: Could not compute P_init"); return None, None, None
    y0 = np.concatenate(([resting_V], P_init))
    try:
        sol = solve_ivp(fun=lambda t, y: ap_ode_rhs(t, y, stim_func, param_dict, Cm, g_K, g_leak, E_K, E_leak, G_Na, E_Na),
                        t_span=(t_eval[0], t_eval[-1]), y0=y0, method='BDF', t_eval=t_eval, rtol=1e-5, atol=1e-7)
        if not sol.success: warnings.warn(f"AP simulation failed: {sol.message}"); return t_eval, None, None
        V_sol = sol.y[0, :]; P_sol = sol.y[1:, :].T
        sum_P = np.sum(P_sol, axis=1, keepdims=True); sum_P[sum_P < NUMERICAL_EPS] = 1.0
        P_sol = P_sol / sum_P; P_sol = np.clip(P_sol, 0.0, 1.0)
        return t_eval, V_sol, P_sol
    except Exception as e: warnings.warn(f"Exception during AP simulation: {e}"); return t_eval, None, None

# --- Computation of Model Discrepancy ---

def compute_protocol_discrepancy(param_dict: Dict[str, float], target_param_dict: Dict[str, float],
                                 protocols: List[str] = ['ssa', 'ssi', 'rec'],
                                 voltages: np.ndarray = DEFAULT_VOLTAGES,
                                 rec_weight: float = 1.0, failure_penalty: float = 1000.0
                                 ) -> float:
    """Compute sum-of-squares discrepancy between model and target."""
    total_discrepancy = 0.0; protocols_computed = 0
    # SSA/SSI
    if 'ssa' in protocols or 'ssi' in protocols:
        ssa_model, ssi_model = compute_SSA_SSI(voltages, param_dict)
        ssa_target, ssi_target = compute_SSA_SSI(voltages, target_param_dict)
        avail_model = 1.0 - ssi_model; avail_target = 1.0 - ssi_target
        valid_ssa = ~np.isnan(ssa_model) & ~np.isnan(ssa_target)
        valid_avail = ~np.isnan(avail_model) & ~np.isnan(avail_target)
        if 'ssa' in protocols:
            if np.sum(valid_ssa) > 0: total_discrepancy += np.sum((ssa_model[valid_ssa] - ssa_target[valid_ssa])**2); protocols_computed += 1
            else: total_discrepancy += failure_penalty
        if 'ssi' in protocols:
             if np.sum(valid_avail) > 0: total_discrepancy += np.sum((avail_model[valid_avail] - avail_target[valid_avail])**2); protocols_computed += 1 if 'ssa' not in protocols else 0
             else: total_discrepancy += failure_penalty
    # Recovery
    if 'rec' in protocols:
        intervals_model, rec_model = simulate_recovery_protocol(param_dict)
        intervals_target, rec_target = simulate_recovery_protocol(target_param_dict)
        rec_target_interp = np.full_like(intervals_model, np.nan)
        valid_target_rec = ~np.isnan(rec_target)
        if np.sum(valid_target_rec) > 1:
            try:
                 interp_func = interp1d(intervals_target[valid_target_rec], rec_target[valid_target_rec], kind='linear', bounds_error=False, fill_value=np.nan)
                 rec_target_interp = interp_func(intervals_model)
            except ValueError: pass
        valid_rec_comp = ~np.isnan(rec_model) & ~np.isnan(rec_target_interp)
        if np.sum(valid_rec_comp) > 0: total_discrepancy += np.sum((rec_model[valid_rec_comp] - rec_target_interp[valid_rec_comp])**2) * rec_weight; protocols_computed += 1
        else: total_discrepancy += failure_penalty
    # Final check
    if protocols_computed == 0 and len(protocols) > 0: return failure_penalty * len(protocols)
    return total_discrepancy

# --- Latin Hypercube Sampling ---

def custom_lhs(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """Basic Latin Hypercube Sampling."""
    rng = np.random.default_rng(seed); samples = np.zeros((n, d))
    for i in range(d): perm = rng.permutation(n); samples[:, i] = (perm + rng.uniform(0, 1, n)) / n
    return samples

def compute_min_distance(design: np.ndarray) -> float:
    """Compute minimum Euclidean distance between any two points."""
    n = design.shape[0];
    if n <= 1: return np.inf
    min_dist_sq = np.inf
    for i in range(n):
        for j in range(i + 1, n): min_dist_sq = min(min_dist_sq, np.sum((design[i] - design[j])**2))
    return np.sqrt(min_dist_sq)

def maximin_lhs(n: int, d: int, n_candidates: int = 100, seed: Optional[int] = None) -> np.ndarray:
    """Generate Latin Hypercube design optimizing maximin distance."""
    rng = np.random.default_rng(seed); best_design = None; max_min_distance = -np.inf
    for i in range(n_candidates):
        candidate_seed = rng.integers(0, 100000); design = custom_lhs(n, d, seed=candidate_seed)
        min_dist = compute_min_distance(design)
        if min_dist > max_min_distance: max_min_distance = min_dist; best_design = design.copy()
    if best_design is None: return np.empty((n, d))
    return best_design

# --- Gaussian Process Metamodeling ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def create_metamodel(X_train: np.ndarray, y_train: np.ndarray
                     ) -> Tuple[Optional[Any], Optional[StandardScaler]]:
    """Create and fit Gaussian Process metamodel with scaling."""
    if X_train.shape[0] < 3: print("Warning: Need >= 3 points for GP."); return None, None
    if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isinf(X_train).any() or np.isinf(y_train).any():
        print("Warning: Training data contains NaN/inf."); return None, None
    if np.any(np.std(X_train, axis=0) < NUMERICAL_EPS): warnings.warn("Input dimensions have low variance.")
    if np.std(y_train) < NUMERICAL_EPS:
        warnings.warn("Output values nearly constant. Adding noise for GP stability.")
        y_train = y_train + np.random.normal(0, 1e-4 * np.abs(np.mean(y_train)), y_train.shape)

    scaler = StandardScaler().fit(X_train); X_train_scaled = scaler.transform(X_train)
    y_mean = np.mean(y_train); y_std = np.std(y_train)
    if y_std < NUMERICAL_EPS: y_std = 1.0
    y_train_scaled = (y_train - y_mean) / y_std

    length_scale_init = np.ones(X_train.shape[1]); length_scale_bounds = (1e-2, 1e2)
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=length_scale_init, length_scale_bounds=length_scale_bounds, nu=2.5) \
           + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False, random_state=SEED, alpha=1e-8)

    try:
        gp.fit(X_train_scaled, y_train_scaled)
        print(f"GP Metamodel trained. Final kernel: {gp.kernel_}")
        class ScaledGP:
            def __init__(self, base_gp, scaler, y_mean, y_std):
                self.base_gp=base_gp; self.scaler=scaler; self.y_mean=y_mean; self.y_std=y_std
            def predict(self, X, return_std=False):
                X_scaled = self.scaler.transform(X)
                mean_scaled, std_scaled = self.base_gp.predict(X_scaled, return_std=True)
                mean = mean_scaled * self.y_std + self.y_mean
                std = std_scaled * self.y_std
                if return_std: return mean, np.maximum(std, NUMERICAL_EPS)
                return mean
        return ScaledGP(gp, scaler, y_mean, y_std), scaler
    except Exception as e: print(f"Error fitting GP model: {e}"); return None, None

def probability_of_improvement(X_eval: np.ndarray, gp: Any, y_best: float, xi: float = 0.01) -> np.ndarray:
    """POI acquisition function."""
    if gp is None: return np.zeros(X_eval.shape[0])
    mu, sigma = gp.predict(X_eval, return_std=True); sigma = np.maximum(sigma, NUMERICAL_EPS)
    improvement = y_best - mu - xi; Z = improvement / sigma
    return norm.cdf(Z)

def find_next_point_poi(gp: Any, param_names: List[str], param_defs: Dict[str, Dict[str, Any]],
                        y_best: float, current_best_x: Optional[np.ndarray] = None,
                        n_restarts: int = 10) -> Optional[np.ndarray]:
    """Find the next evaluation point by maximizing POI."""
    if gp is None: return None
    bounds_list = [(param_defs[name]['low'], param_defs[name]['high']) for name in param_names]
    bounds_arr = np.array(bounds_list)
    def neg_poi_obj(x):
        try: poi_val = probability_of_improvement(x.reshape(1, -1), gp, y_best)[0]
        except Exception: return 1.0
        return -poi_val if np.isfinite(poi_val) else 1.0

    best_x_next = None; min_neg_poi = np.inf
    start_points = []
    lhs_starts = custom_lhs(n_restarts * 2, len(param_names)); scaled_starts = bounds_arr[:, 0] + lhs_starts * (bounds_arr[:, 1] - bounds_arr[:, 0])
    start_points.extend(list(scaled_starts))
    if current_best_x is not None: start_points.append(current_best_x)

    for i, x0 in enumerate(start_points):
        try:
            res = minimize(neg_poi_obj, x0, method='L-BFGS-B', bounds=bounds_list)
            if res.success and res.fun < min_neg_poi:
                min_neg_poi = res.fun; best_x_next = np.clip(res.x, bounds_arr[:, 0], bounds_arr[:, 1])
        except Exception as e: warnings.warn(f"POI minimize failed for start {i}: {e}")

    if best_x_next is None:
        warnings.warn("POI optimization failed. Selecting random point.");
        best_x_next = bounds_arr[:, 0] + np.random.random(len(param_names)) * (bounds_arr[:, 1] - bounds_arr[:, 0])
    return best_x_next


# --- Parallel Evaluation to Speed up (a little bit...)---

def _evaluate_single_point(x_opt: np.ndarray, param_names: List[str],
                           p_nominal_full: Dict[str, float], p_target: Dict[str, float],
                           protocols: List[str], voltages: np.ndarray) -> Optional[float]:
    """
    Evaluates discrepancy for a single parameter vector 'x_opt'.
    Returns discrepancy value or np.nan upon failure.
    """
    try:
        p_run = p_nominal_full.copy()
        for k, name in enumerate(param_names):
            p_run[name] = x_opt[k]
        p_run = _update_dependent_params(p_run)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            disc = compute_protocol_discrepancy(p_run, p_target, protocols, voltages)

        # Return NaN if discrepancy is not finite
        return disc if np.isfinite(disc) else np.nan
    except Exception as e:
        # Log the error maybe? For now, just return NaN on any exception
        print(f"Warning: Error evaluating point {x_opt}: {e}")
        return np.nan


# --- Sequential Optimization with Parallel Initial Evaluation (still very slow)---

def sequential_optimization(param_names: List[str], param_defs: Dict[str, Dict[str, Any]],
                            p_nominal_full: Dict[str, float], p_target: Dict[str, float],
                            protocols: List[str], voltages: np.ndarray,
                            n_initial: int, n_iterations: int, tol: float,
                            n_poi_restarts: int = 10, convergence_patience: int = 5
                            ) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Perform sequential optimization using GP metamodeling and POI, with parallel initial eval."""
    n_dim = len(param_names)
    print(f"\n--- Starting Sequential Optimization ---")
    print(f"  Parameters ({n_dim}): {', '.join(param_names)}")
    print(f"  Protocols: {', '.join(protocols)}")
    print(f"  Target tolerance: {tol:.4g}")
    print(f"  Initial points: {n_initial}, Max iterations: {n_iterations}")

    bounds_arr = np.array([(param_defs[name]['low'], param_defs[name]['high']) for name in param_names])
    lower_bounds = bounds_arr[:, 0]; upper_bounds = bounds_arr[:, 1]

    print(f"Generating {n_initial} initial design points using Maximin LHS...")
    X_initial_01 = maximin_lhs(n_initial, n_dim, n_candidates=n_initial * 10, seed=SEED)
    X_initial_scaled = lower_bounds + X_initial_01 * (upper_bounds - lower_bounds)

    X_hist_list = []
    y_hist_list = []

    # --- Parallel Evaluation of Initial Points ---
    print(f"Evaluating {n_initial} initial points in parallel...")
    initial_eval_start_time = time.time()

    # Create the partial function with fixed arguments for the evaluator
    partial_evaluate_func = partial(_evaluate_single_point,
                                    param_names=param_names,
                                    p_nominal_full=p_nominal_full.copy(),
                                    p_target=p_target.copy(),
                                    protocols=protocols,
                                    voltages=voltages)

    try:
        n_workers = cpu_count()
        print(f"Using {n_workers} worker processes.")
    except NotImplementedError:
        n_workers = 1
        print("cpu_count() not implemented, using 1 worker process.")

    try:
        with Pool(processes=n_workers) as pool:
            results = pool.map(partial_evaluate_func, list(X_initial_scaled)) # Pass list
    except Exception as e:
        print(f"Error during parallel processing: {e}. Aborting.")
        return p_nominal_full, {'X': [], 'y': [], 'best_y_so_far': [], 'param_names': param_names, 'error': 'Parallel processing failed'}

    # Process results
    points_evaluated = 0
    for x_opt, disc in zip(X_initial_scaled, results):
        if disc is not None and not np.isnan(disc):
            X_hist_list.append(x_opt)
            y_hist_list.append(disc)
            points_evaluated += 1
        else:
            print(f"  Warning: Skipping initial point evaluation result ({disc}).")


    initial_eval_duration = time.time() - initial_eval_start_time
    print(f"Initial parallel evaluation complete ({points_evaluated} valid points) in {initial_eval_duration:.2f}s.")

    # --- Continue with Optimization ---
    if points_evaluated < 3:
        print("Error: Failed to evaluate enough initial points to start GP modeling.")
        return p_nominal_full, {'X': [], 'y': [], 'best_y_so_far': [], 'param_names': param_names, 'error': 'Insufficient initial points'}

    X_hist = np.array(X_hist_list); y_hist = np.array(y_hist_list)
    best_idx_init = np.argmin(y_hist); y_best = y_hist[best_idx_init]; x_best = X_hist[best_idx_init].copy()
    print(f"Initial best discrepancy: {y_best:.5f}")

    history = {'X': X_hist.tolist(), 'y': y_hist.tolist(), 'best_y_so_far': [y_best],
               'param_names': param_names, 'iterations_run': 0, 'convergence_reason': None}

    no_improvement_counter = 0; optimization_start_time = time.time()

    for i in range(n_iterations):
        iter_start_time = time.time(); history['iterations_run'] = i + 1
        print(f"\n--- Optimization Iteration {i + 1}/{n_iterations} (Current best: {y_best:.5f}) ---")

        if y_best <= tol: print(f"Convergence tolerance {tol:.4g} reached."); history['convergence_reason'] = 'Tolerance reached'; break
        if no_improvement_counter >= convergence_patience: print(f"Convergence: No improvement in {convergence_patience} iterations."); history['convergence_reason'] = f'Stagnation ({convergence_patience} iterations)'; break

        print("Fitting GP metamodel..."); gp_fit_start = time.time()
        gp, _ = create_metamodel(X_hist, y_hist)
        gp_fit_duration = time.time() - gp_fit_start; print(f"GP fitting took {gp_fit_duration:.2f}s")
        if gp is None: print("Warning: GP fitting failed."); history['convergence_reason'] = 'GP fitting failed'; break

        print("Finding next point using POI..."); poi_start = time.time()
        x_next = find_next_point_poi(gp, param_names, param_defs, y_best, x_best, n_poi_restarts)
        poi_duration = time.time() - poi_start; print(f"POI optimization took {poi_duration:.2f}s")
        if x_next is None: print("Warning: Failed to find next point."); history['convergence_reason'] = 'POI optimization failed'; break

        print(f"Evaluating next point: {dict(zip(param_names, np.round(x_next, 4)))}")
        eval_start = time.time()
        # Evaluate the single next point
        y_next = _evaluate_single_point(x_next, param_names, p_nominal_full.copy(), p_target.copy(), protocols, voltages)
        eval_duration = time.time() - eval_start

        if y_next is not None and not np.isnan(y_next):
            print(f"Evaluation took {eval_duration:.2f}s. Discrepancy = {y_next:.5f}")
            X_hist = np.vstack((X_hist, x_next)); y_hist = np.append(y_hist, y_next)
            history['X'].append(x_next.tolist()); history['y'].append(y_next)
            if y_next < y_best - (NUMERICAL_EPS * abs(y_best)):
                print(f"*** New best discrepancy found: {y_next:.5f} ***")
                y_best = y_next; x_best = x_next.copy(); no_improvement_counter = 0
            else: no_improvement_counter += 1
            history['best_y_so_far'].append(y_best)
        else:
            print(f"Warning: Invalid discrepancy ({y_next}) for the next point. Skipping update.")
            no_improvement_counter += 1

        iter_duration = time.time() - iter_start_time
        print(f"Iteration {i+1} finished in {iter_duration:.2f}s. Stagnation counter: {no_improvement_counter}")

    # --- End of Loop ---
    optimization_duration = time.time() - optimization_start_time
    print(f"\nOptimization loop finished after {history['iterations_run']} iterations in {optimization_duration:.2f}s.")
    if history['convergence_reason'] is None: history['convergence_reason'] = 'Max iterations reached'; print("Max iterations reached.")

    p_optimal = p_nominal_full.copy()
    if x_best is not None:
         for k, name in enumerate(param_names): p_optimal[name] = x_best[k]
         p_optimal = _update_dependent_params(p_optimal)
    else: print("Warning: No valid 'best' parameters found."); p_optimal = p_nominal_full
    print(f"Final best discrepancy: {y_best:.5f}")
    return p_optimal, history


# --- Plotting Functions ---
def plot_SSA_SSI_comparison(voltages: np.ndarray, models_dict: Dict[str, Dict[str, float]],
                            title: str = "SSA & Availability Comparison") -> plt.Figure:
    """Plots SSA and Availability (1-SSI) for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', '*', 'p']
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_dict))) # Viridis is colorblind-friendly

    for i, (name, p_dict) in enumerate(models_dict.items()):
        try:
            ssa, ssi = compute_SSA_SSI(voltages, p_dict)
            availability = 1.0 - ssi
            valid_ssa = ~np.isnan(ssa)
            valid_avail = ~np.isnan(availability)
            m = markers[i % len(markers)]
            c = colors[i]

            ax.plot(voltages[valid_ssa], ssa[valid_ssa], '-', color=c,
                    label=f'{name} SSA', linewidth=2, marker=m, markersize=5, markevery=2)
            ax.plot(voltages[valid_avail], availability[valid_avail], '--', color=c,
                    label=f'{name} Availability', linewidth=2, marker=m, markersize=5, markevery=2)
        except Exception as e:
            print(f"Warning: Could not plot SSA/SSI for model '{name}': {e}")

    ax.set_xlabel('Voltage (mV)', fontsize=12)
    ax.set_ylabel('Probability / Availability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=False)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    return fig

def plot_recovery_comparison(models_dict: Dict[str, Dict[str, float]],
                             title: str = "Recovery from Fast Inactivation"
                             ) -> Tuple[plt.Figure, Dict[str, Dict[str, Any]]]:
    """Plots recovery curves and calculates RP50 for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', '*', 'p']
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_dict)))
    results = {}

    for i, (name, p_dict) in enumerate(models_dict.items()):
        try:
            intervals, recovery = simulate_recovery_protocol(p_dict)
            rp50 = calculate_refractory_period(intervals, recovery)
            results[name] = {'intervals': intervals, 'recovery': recovery, 'rp50': rp50}
            valid_rec = ~np.isnan(recovery)
            label = f'{name} (RP50: {rp50:.1f} ms)' if not np.isnan(rp50) else f'{name} (RP50: N/A)'
            m = markers[i % len(markers)]
            c = colors[i]
            ax.plot(intervals[valid_rec], recovery[valid_rec], '-', color=c, label=label,
                    linewidth=2, marker=m, markersize=5, markevery=5)
        except Exception as e:
            print(f"Warning: Could not plot recovery for model '{name}': {e}")
            results[name] = {'intervals': None, 'recovery': None, 'rp50': np.nan}


    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, label='50% Recovery')
    ax.set_xlabel('Recovery Interval (ms)', fontsize=12)
    ax.set_ylabel('Fraction Recovered (Normalized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=False)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    plt.tight_layout()
    return fig, results


def plot_optimization_history(history: Dict[str, Any], title: str = 'Optimization History') -> plt.Figure:
    """Plots the optimization convergence history."""
    if not history or not history.get('y'):
        print("Warning: Cannot plot empty optimization history.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No history data to plot", ha='center', va='center')
        return fig

    param_names = history.get('param_names', [])
    y_evals = np.array(history['y'])
    n_evals = len(y_evals)
    iterations = np.arange(n_evals)

    # Calculate running best
    best_y_so_far = np.minimum.accumulate(y_evals)

    # Determine initial design phase end
    iterations_run = history.get('iterations_run', n_evals - 1) # Iterations *after* initial
    n_initial = n_evals - iterations_run

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot evaluated points
    ax1.plot(iterations, y_evals, 'o', color='steelblue', alpha=0.5, markersize=5, label='Evaluated Discrepancy')
    # Plot best found so far
    ax1.plot(iterations, best_y_so_far, '-', color='red', lw=2.5, label='Best Discrepancy Found')

    # Mark end of initial design
    if n_initial > 0 and n_initial < n_evals:
        ax1.axvline(n_initial - 0.5, color='black', linestyle='--', lw=1.5, label='End of Initial Design')

    ax1.set_xlabel('Evaluation Number', fontsize=12)
    ax1.set_ylabel('Discrepancy (Sum of Squares)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, frameon=False)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax1.set_xlim(left=-1)

    if param_names and 0 < len(param_names) <= 10:
        ax2 = ax1.twinx()
        X_array = np.array(history['X'])
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))

        for i, name in enumerate(param_names):
            param_vals = X_array[:, i]
            mean_val = np.mean(param_vals)
            std_val = np.std(param_vals)
            if std_val > NUMERICAL_EPS:
                 scaled_vals = (param_vals - mean_val) / std_val
            else:
                 scaled_vals = np.zeros_like(param_vals)
            ax2.plot(iterations, scaled_vals, '--', color=colors[i], lw=1, alpha=0.7, label=f'{name} (scaled)')

        ax2.set_ylabel('Scaled Parameter Value', fontsize=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.legend(loc='lower right', fontsize=8, frameon=False, ncol=2)
        ax2.grid(False)


    plt.tight_layout()
    return fig


def visualize_state_transitions(param_dict: Dict[str, float],
                                time_points: List[float] = [11, 12, 28, 50, 100],
                                figsize: Tuple[int, int] = (12, 10)
                                ) -> Optional[plt.Figure]:
    """Visualize state transitions similar to Figure 13 in the paper."""
    print(f"  Simulating AP for state transition viz...")
    duration = max(time_points) + 50
    t, V, P = simulate_action_potential(param_dict, duration)

    if V is None or P is None:
        print("  Action potential simulation failed. Cannot visualize states.")
        return None

    if np.isnan(V).any():
        print("  Warning: NaN values detected in voltage trace. Visualization might be affected.")
        V = np.nan_to_num(V, nan=RESTING_V_DEFAULT)


    # --- Create Figure ---
    fig = plt.figure(figsize=figsize)
    n_rows = len(time_points) + 1
    gs = GridSpec(n_rows, 2, width_ratios=[1, 2.5],
                  height_ratios=[1.5] + [1] * len(time_points),
                  hspace=0.4, wspace=0.1)

    # --- Plot Full Action Potential ---
    ax_ap_full = fig.add_subplot(gs[0, :])
    ax_ap_full.plot(t, V, 'k-', linewidth=1.5)
    for tp in time_points:
        ax_ap_full.axvline(tp, color='red', linestyle=':', alpha=0.6, lw=1)
        try:
            idx = np.nanargmin(np.abs(t - tp))
            ax_ap_full.plot(tp, V[idx], 'ro', markersize=4, alpha=0.8)
        except (ValueError, IndexError): pass

    ax_ap_full.set_xlabel('Time (ms)')
    ax_ap_full.set_ylabel('Voltage (mV)')
    ax_ap_full.set_title('Action Potential Trace')
    ax_ap_full.grid(True, linestyle=':', alpha=0.5)
    ax_ap_full.set_xlim(left=0, right=duration)

    # --- Plot State Distributions at each Time Point ---
    node_pos = {
        'IC3': (0, 1), 'IC2': (1, 1), 'IF': (2, 1), 'I1': (3, 1), 'I2': (4, 1),
        'C3': (0, 0), 'C2': (1, 0), 'C1': (2, 0), 'O': (3, 0)
    }
    x_coords = {name: pos[0] for name, pos in node_pos.items()}
    y_coords = {name: pos[1] for name, pos in node_pos.items()}
    max_x = max(x_coords.values())
    max_y = max(y_coords.values())

    for i, tp in enumerate(time_points):
        try: # Find index safely
            idx = np.nanargmin(np.abs(t - tp))
            state_probs = P[idx]
            current_V = V[idx]
        except (ValueError, IndexError):
            state_probs = np.zeros(N_STATES)
            state_probs[STATE_LABELS.index('C3')] = 1.0
            current_V = np.nan

        # Left panel: AP snippet
        ax_ap_t = fig.add_subplot(gs[i + 1, 0])
        ax_ap_t.plot(t, V, 'k-', linewidth=1, alpha=0.3)
        ax_ap_t.axvline(tp, color='red', linestyle='--')
        if not np.isnan(current_V):
             ax_ap_t.plot(tp, current_V, 'ro', markersize=5)
        ax_ap_t.text(0.05, 0.9, f't = {tp} ms', transform=ax_ap_t.transAxes,
                     fontsize=10, fontweight='bold', ha='left', va='top')
        ax_ap_t.set_xlim(max(0, tp - 20), min(duration, tp + 20)) # Show local window
        ax_ap_t.set_xticks([])
        ax_ap_t.set_yticks([])
        ax_ap_t.axis('off')

        # Right panel: State probability visualization
        ax_states = fig.add_subplot(gs[i + 1, 1])
        min_radius = 0.05
        max_radius = 0.4
        for s, idx_s in enumerate(STATE_LABELS):
            prob = np.clip(state_probs[s], 0, 1)
            x, y = node_pos[STATE_LABELS[s]]
            radius = min_radius + (max_radius - min_radius) * np.sqrt(prob)
            color_intensity = 0.1 + 0.9 * prob

            circle = plt.Circle((x, y), radius,
                                color=plt.cm.Blues(color_intensity),
                                alpha=0.8)
            ax_states.add_patch(circle)
            if radius > 0.15:
                ax_states.text(x, y, f"{prob:.2f}", ha='center', va='center',
                               fontsize=8 if radius > 0.3 else 7,
                               color='white' if color_intensity > 0.5 else 'black')
            ax_states.text(x, y - max_radius * 1.5, STATE_LABELS[s], ha='center', va='top', fontsize=9)

        connections = [('C3','C2'), ('C2','C1'), ('C1','O'), ('C1','IF'),
                       ('IC3','C3'), ('IC2','C2'), ('IF','IC2'), ('IF','I1'),
                       ('I1','I2'), ('IC3','IC2'), ('I1','O')]
        for u, v in connections:
            if u in node_pos and v in node_pos:
                 xu, yu = node_pos[u]
                 xv, yv = node_pos[v]
                 ax_states.plot([xu, xv], [yu, yv], '-', color='gray', lw=0.5, alpha=0.5, zorder=-1)


        ax_states.set_xlim(-max_radius*2, max_x + max_radius*2)
        ax_states.set_ylim(-max_radius*2, max_y + max_radius*2)
        ax_states.set_aspect('equal', adjustable='box')
        ax_states.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"State Probability Distributions at Key Time Points", fontsize=16, fontweight='bold')
    return fig


# --- Main Execution  ---

def optimize_sequentially(target_condition='KO', param_variation=0.5,
                          n_init_factor=2, n_iter_ssa=30, n_iter_ssi_rec=50,
                          tol_ssa=0.004, tol_ssi_rec=0.04):
    """Runs the two-stage sequential optimization (SSA then SSI+REC) with parallel initial eval."""
    print(f"\n=== Starting Sequential Optimization Workflow ===")
    print(f"Target Condition: {target_condition}")
    print(f"Parameter Bounds: WT nominal +/- {param_variation*100:.0f}%")

    p_WT = get_parameters_WT()
    p_target = get_parameters_KO() if target_condition == 'KO' else get_parameters_WT()
    p_nominal_start = p_WT
    param_defs = get_parameter_definitions(variation=param_variation)
    voltages_eval = DEFAULT_VOLTAGES
    overall_start_time = time.time()

    # --- Stage 1: Optimize SSA parameters ---
    print("\n=== STAGE 1: SSA Parameter Optimization ===")
    n_initial_ssa = max(10, n_init_factor * len(SSA_IMPORTANT_PARAMS))
    p_ssa_optimal, ssa_history = sequential_optimization(
        param_names=SSA_IMPORTANT_PARAMS, param_defs=param_defs,
        p_nominal_full=p_nominal_start, p_target=p_target, protocols=['ssa'],
        voltages=voltages_eval, n_initial=n_initial_ssa, n_iterations=n_iter_ssa, tol=tol_ssa
    )

    # --- Stage 2: Optimize SSI and REC parameters ---
    print("\n=== STAGE 2: SSI and REC Parameter Optimization ===")
    n_initial_ssi_rec = max(10, n_init_factor * len(SSI_REC_IMPORTANT_PARAMS))
    p_start_stage2 = p_nominal_start.copy()
    p_start_stage2.update(p_ssa_optimal)
    p_start_stage2 = _update_dependent_params(p_start_stage2)
    p_final_optimal, ssi_rec_history = sequential_optimization(
        param_names=SSI_REC_IMPORTANT_PARAMS, param_defs=param_defs,
        p_nominal_full=p_start_stage2, p_target=p_target, protocols=['ssi', 'rec'],
        voltages=voltages_eval, n_initial=n_initial_ssi_rec, n_iterations=n_iter_ssi_rec, tol=tol_ssi_rec
    )

    # Combine results
    p_combined_optimal = p_ssa_optimal.copy()
    p_combined_optimal.update(p_final_optimal)
    for name in FIXED_PARAMS_VALUES: p_combined_optimal[name] = FIXED_PARAMS_VALUES[name]
    p_combined_optimal = _update_dependent_params(p_combined_optimal)

    overall_end_time = time.time()
    print(f"\n=== Sequential Optimization Workflow Complete ===")
    print(f"Total time: {overall_end_time - overall_start_time:.2f} seconds.")

    # Save and Analyze Results
    results_package = {'parameters': {'WT': p_WT, 'Target': p_target, 'Optimized_SSA_Stage': p_ssa_optimal, 'Optimized_Final': p_combined_optimal},
                       'history': {'SSA': ssa_history, 'SSI_REC': ssi_rec_history},
                       'config': {'target_condition': target_condition, 'param_variation': param_variation, 'n_init_factor': n_init_factor,
                                  'n_iter_ssa': n_iter_ssa, 'n_iter_ssi_rec': n_iter_ssi_rec, 'tol_ssa': tol_ssa, 'tol_ssi_rec': tol_ssi_rec, 'seed': SEED}}
    timestamp = time.strftime("%Y%m%d_%H%M%S"); results_filename = f"sequential_opt_results_{target_condition}_{timestamp}.pkl"
    try:
        with open(results_filename, 'wb') as f: pickle.dump(results_package, f)
        print(f"Results saved to {results_filename}")
    except Exception as e: print(f"Error saving results: {e}")

    # Post-Optimization Analysis & Plotting
    print("\n--- Generating Comparison Plots ---")
    models_to_compare = {'WT': p_WT, f'{target_condition} (Target)': p_target, 'Optimized': p_combined_optimal}
    try:
        print("Plotting SSA and Availability..."); fig_ssa_ssi = plot_SSA_SSI_comparison(voltages_eval, models_to_compare, title=f"SSA & Availability (Optimized vs {target_condition})"); fig_ssa_ssi.savefig(f"results_ssa_ssi_comparison_{target_condition}_{timestamp}.png", dpi=300); plt.close(fig_ssa_ssi)
        print("Plotting Recovery..."); fig_rec, recovery_results = plot_recovery_comparison(models_to_compare, title=f"Recovery (Optimized vs {target_condition})"); fig_rec.savefig(f"results_recovery_comparison_{target_condition}_{timestamp}.png", dpi=300); plt.close(fig_rec)
        print("\nRefractory Periods (RP50):")
        for name, data in recovery_results.items(): rp = data.get('rp50', np.nan); print(f"  {name:<18}: {rp:.2f} ms" if not np.isnan(rp) else f"  {name:<18}: N/A")
        print("Plotting Optimization History..."); fig_ssa_opt = plot_optimization_history(ssa_history, title="SSA Optimization Convergence"); fig_ssa_opt.savefig(f"results_ssa_optimization_{target_condition}_{timestamp}.png", dpi=300); plt.close(fig_ssa_opt)
        fig_ssi_rec_opt = plot_optimization_history(ssi_rec_history, title="SSI+REC Optimization Convergence"); fig_ssi_rec_opt.savefig(f"results_ssi_rec_optimization_{target_condition}_{timestamp}.png", dpi=300); plt.close(fig_ssi_rec_opt)
        print("Visualizing state transitions (this may take time)...")
        print("   (State transition plots skipped in this example run to save time)")

    except Exception as e: print(f"Error during plotting or final analysis: {e}")

    print("\n--- Final Parameter Comparison ---"); all_optimized_params = sorted(list(set(SSA_IMPORTANT_PARAMS + SSI_REC_IMPORTANT_PARAMS)))
    print(f"{'Parameter':<10} {'WT':>14} {f'{target_condition} (Target)':>18} {'Optimized':>14}"); print("-" * 60)
    for param in all_optimized_params: wt_val = p_WT.get(param, np.nan); target_val = p_target.get(param, np.nan); opt_val = p_combined_optimal.get(param, np.nan); print(f"{param:<10} {wt_val:>14.4f} {target_val:>18.4f} {opt_val:>14.4f}")
    print("\n=== Workflow Finished ===")
    return results_package


if __name__ == "__main__":
    # Set target condition ('KO' or 'WT')
    TARGET_CONDITION = 'KO'

    # Run the optimization workflow
    optimization_results = optimize_sequentially(
        target_condition=TARGET_CONDITION,
        param_variation=0.5,
        n_init_factor=2,
        n_iter_ssa=20,       # Reduced for testing
        n_iter_ssi_rec=30,   # Reduced for testing
        tol_ssa=0.004,
        tol_ssi_rec=0.04
    )

    print("\n--- Main Script Finished ---")