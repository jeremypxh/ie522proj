import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time

from scm import get_parameters_WT, get_parameters_KO, compute_SSA_SSI
from scm import get_stationary_distribution, simulate_AP, simulate_channel_dynamics
from scm import plot_state_distributions, compute_current_density

# Define voltage ranges for different experiments
voltages_fine = np.arange(-140, 21, 2)
voltages_coarse = np.arange(-140, 21, 10)  # For optimization/screening


# --------------------------------
# 1. Parameter Screening Functions
# --------------------------------

def create_factorial_design(param_names, levels, fraction=0):
    """
    Create a fractional factorial design for parameter screening.

    Parameters:
    -----------
    param_names : list
        List of parameter names to vary
    levels : dict
        Dictionary with (min, max) values for each parameter
    fraction : int
        Power of 1/2 for the fraction (0 for full factorial)

    Returns:
    --------
    design_matrix : numpy array
        Design matrix with -1/+1 levels
    param_values : list of dict
        List of parameter dictionaries for each run
    """
    n_params = len(param_names)
    n_runs = 2 ** (n_params - fraction)

    # Create full factorial design in -1/+1 coding
    if fraction == 0:
        # Full factorial design
        design_matrix = np.zeros((n_runs, n_params))
        for i in range(n_runs):
            for j in range(n_params):
                design_matrix[i, j] = -1 if (i // 2 ** j) % 2 == 0 else 1
    else:
        # Basic implementation of fractional factorial
        # This is simplified - a more sophisticated approach would use generators
        base_params = n_params - fraction
        # Create full factorial for base parameters
        base_design = np.zeros((2 ** base_params, base_params))
        for i in range(2 ** base_params):
            for j in range(base_params):
                base_design[i, j] = -1 if (i // 2 ** j) % 2 == 0 else 1

        # Add additional columns as interactions for the fraction
        design_matrix = np.zeros((2 ** base_params, n_params))
        design_matrix[:, :base_params] = base_design

        # Generate additional columns as interactions (simplified approach)
        for j in range(base_params, n_params):
            # Use two-factor interactions as generators (can be customized)
            col1 = j % base_params
            col2 = (j + 1) % base_params
            design_matrix[:, j] = base_design[:, col1] * base_design[:, col2]

    # Convert design matrix to parameter values
    param_values = []
    for i in range(n_runs):
        run_params = {}
        for j, param in enumerate(param_names):
            low, high = levels[param]
            if design_matrix[i, j] == -1:
                run_params[param] = low
            else:
                run_params[param] = high
        param_values.append(run_params)

    return design_matrix, param_values


def compute_factorial_effects(design_matrix, responses, param_names):
    """
    Compute main effects for each parameter from factorial experiment results.

    Parameters:
    -----------
    design_matrix : numpy array
        Design matrix with -1/+1 levels
    responses : numpy array
        Response values for each run
    param_names : list
        List of parameter names

    Returns:
    --------
    effects : dict
        Dictionary of main effects for each parameter
    """
    n_runs, n_params = design_matrix.shape
    effects = {}

    for j, param in enumerate(param_names):
        # Compute contrast for this parameter
        contrast = 0
        for i in range(n_runs):
            contrast += design_matrix[i, j] * responses[i]

        # Divide by number of runs and multiply by 2
        effects[param] = (2.0 / n_runs) * contrast

    return effects


def plot_half_normal(effects, title='Half-Normal Plot of Effects'):
    """
    Create a half-normal probability plot of parameter effects.

    Parameters:
    -----------
    effects : dict
        Dictionary of main effects for each parameter
    title : str
        Plot title

    Returns:
    --------
    fig : matplotlib figure
        The generated figure
    """
    # Extract absolute values of effects
    param_names = list(effects.keys())
    abs_effects = np.abs([effects[p] for p in param_names])

    # Sort by absolute effect size
    sorted_indices = np.argsort(abs_effects)
    sorted_params = [param_names[i] for i in sorted_indices]
    sorted_effects = abs_effects[sorted_indices]

    # Generate theoretical quantiles
    n = len(sorted_effects)
    probs = (np.arange(1, n + 1) - 0.5) / n
    quantiles = norm.ppf(0.5 + probs / 2)  # Half-normal distribution

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(quantiles, sorted_effects, 'o', markersize=6)

    # Add parameter labels
    for i, param in enumerate(sorted_params):
        ax.annotate(param, (quantiles[i], sorted_effects[i]),
                    xytext=(5, 5), textcoords='offset points')

    # Draw reference line through origin and largest point
    # Uncomment if needed - sometimes reference lines can be misleading
    # max_idx = np.argmax(sorted_effects)
    # if max_idx > 0:  # Ensure at least one point to define the line
    #     slope = sorted_effects[max_idx] / quantiles[max_idx]
    #     ax.plot([0, quantiles[-1]], [0, slope * quantiles[-1]], 'r--')

    ax.set_xlabel('Half-Normal Quantiles')
    ax.set_ylabel('|Effect|')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def screen_parameters(param_dict, param_levels, voltages, n_runs=None, fraction=4):
    """
    Screen parameters to identify those with significant effects on model responses.

    Parameters:
    -----------
    param_dict : dict
        Nominal parameter values
    param_levels : dict
        Dictionary with (min, max) values for each parameter
    voltages : numpy array
        Voltage values for evaluation
    n_runs : int, optional
        Number of runs (default: determined by fraction)
    fraction : int
        Power of 1/2 for the fraction

    Returns:
    --------
    important_params : dict
        Dictionary with important parameters for SSA and SSI
    """
    # Define parameters to vary (excluding constants and derived)
    param_names = list(param_levels.keys())
    print(f"Screening {len(param_names)} parameters using fractional factorial design")

    # Create fractional factorial design
    design_matrix, param_values = create_factorial_design(param_names, param_levels, fraction)
    n_runs = len(param_values)
    print(f"Factorial design with {n_runs} runs created")

    # Evaluate responses for each design point
    ssa_responses = []
    ssi_responses = []

    print("Evaluating design points...")
    start_time = time.time()

    for i, params in enumerate(param_values):
        # Create a complete parameter set by updating nominal values
        run_params = param_dict.copy()
        for param, value in params.items():
            run_params[param] = value

        # Update dependent parameters
        if 'x14' in params:
            run_params['x16'] = run_params['x14']
        if 'x18' in params:
            run_params['x20'] = run_params['x18']
        if 'x19' in params:
            run_params['x21'] = run_params['x19']

        # Compute SSA and SSI
        try:
            ssa, ssi = compute_SSA_SSI(voltages, run_params)

            # Use average values as responses
            ssa_responses.append(np.mean(ssa))
            ssi_responses.append(np.mean(ssi))

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Completed {i + 1}/{n_runs} runs ({elapsed:.1f} seconds)")
        except Exception as e:
            print(f"Error in run {i + 1}: {e}")
            # Assign neutral value if computation fails
            ssa_responses.append(0)
            ssi_responses.append(0)

    # Compute effects
    ssa_effects = compute_factorial_effects(design_matrix, np.array(ssa_responses), param_names)
    ssi_effects = compute_factorial_effects(design_matrix, np.array(ssi_responses), param_names)

    # Plot half-normal plots
    fig_ssa = plot_half_normal(ssa_effects, 'Half-Normal Plot: SSA Effects')
    fig_ssi = plot_half_normal(ssi_effects, 'Half-Normal Plot: SSI Effects')

    # Identify important parameters (simplified approach - could be more sophisticated)
    # For now, just take the top parameters by absolute effect size
    n_important = min(10, len(param_names) // 3)  # Take approximately 1/3 of parameters

    ssa_important = sorted(ssa_effects.items(), key=lambda x: abs(x[1]), reverse=True)[:n_important]
    ssi_important = sorted(ssi_effects.items(), key=lambda x: abs(x[1]), reverse=True)[:n_important]

    ssa_params = [p[0] for p in ssa_important]
    ssi_params = [p[0] for p in ssi_important]

    print("\nImportant parameters for SSA:", ssa_params)
    print("Important parameters for SSI:", ssi_params)

    return {
        'ssa_important': ssa_params,
        'ssi_important': ssi_params,
        'ssa_effects': ssa_effects,
        'ssi_effects': ssi_effects,
        'ssa_fig': fig_ssa,
        'ssi_fig': fig_ssi
    }


# -------------------------------------------
# 2. Latin Hypercube Design and Optimization
# -------------------------------------------

def latin_hypercube_design(n_samples, param_bounds):
    """
    Generate a Latin Hypercube Design with maximin distance criterion.

    Parameters:
    -----------
    n_samples : int
        Number of sample points
    param_bounds : list of tuples
        List of (min, max) bounds for each parameter

    Returns:
    --------
    design : numpy array
        Design matrix with parameter values
    """
    n_params = len(param_bounds)

    # Generate Latin Hypercube samples in [0, 1]
    # Generate random permutations for each dimension
    design = np.zeros((n_samples, n_params))

    for j in range(n_params):
        perm = np.random.permutation(n_samples)
        design[:, j] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples

    # Scale to parameter bounds
    for j, (low, high) in enumerate(param_bounds):
        design[:, j] = low + design[:, j] * (high - low)

    # Apply maximin criterion (simple approach: generate multiple designs and pick the best)
    # This is a simplified version - a full implementation would be more sophisticated
    best_design = design
    best_min_dist = compute_min_distance(design)

    # Try a few more designs to improve minimum distance
    for _ in range(5):
        new_design = np.zeros((n_samples, n_params))
        for j in range(n_params):
            perm = np.random.permutation(n_samples)
            new_design[:, j] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples

        # Scale to parameter bounds
        for j, (low, high) in enumerate(param_bounds):
            new_design[:, j] = low + new_design[:, j] * (high - low)

        new_min_dist = compute_min_distance(new_design)
        if new_min_dist > best_min_dist:
            best_design = new_design
            best_min_dist = new_min_dist

    return best_design


def compute_min_distance(design):
    """
    Compute the minimum distance between any two points in the design.

    Parameters:
    -----------
    design : numpy array
        Design matrix

    Returns:
    --------
    min_dist : float
        Minimum distance
    """
    n_samples = design.shape[0]
    min_dist = float('inf')

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.sqrt(np.sum((design[i] - design[j]) ** 2))
            min_dist = min(min_dist, dist)

    return min_dist


def compute_model_discrepancy(params, target_params, voltages):
    """
    Compute the discrepancy between model with params and target model.

    Parameters:
    -----------
    params : dict
        Parameter values to evaluate
    target_params : dict
        Target parameter values to compare against
    voltages : numpy array
        Voltage values for evaluation

    Returns:
    --------
    discrepancy : float
        Sum of absolute differences in SSA and SSI
    """
    # Compute SSA and SSI for both parameter sets
    ssa_model, ssi_model = compute_SSA_SSI(voltages, params)
    ssa_target, ssi_target = compute_SSA_SSI(voltages, target_params)

    # Compute discrepancy as sum of absolute differences
    disc_ssa = np.mean(np.abs(ssa_model - ssa_target))
    disc_ssi = np.mean(np.abs(ssi_model - ssi_target))

    return disc_ssa + disc_ssi


def update_gaussian_process(X, y):
    """
    Create or update a Gaussian Process model.

    Parameters:
    -----------
    X : numpy array
        Input points
    y : numpy array
        Response values

    Returns:
    --------
    gp : GaussianProcessRegressor
        Fitted GP model
    """
    # Define the kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * X.shape[1],
                                       length_scale_bounds=(1e-2, 1e2))

    # Create and fit the GP model
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-8,
        random_state=42
    )

    gp.fit(X, y)
    return gp


def expected_improvement(gp, X, best_y, xi=0.01):
    """
    Compute expected improvement acquisition function.

    Parameters:
    -----------
    gp : GaussianProcessRegressor
        Fitted GP model
    X : numpy array
        Points at which to evaluate EI
    best_y : float
        Current best function value
    xi : float
        Exploration parameter

    Returns:
    --------
    ei : numpy array
        Expected improvement values
    """
    mu, sigma = gp.predict(X, return_std=True)

    # Ensure positive standard deviation (avoid numerical issues)
    sigma = np.maximum(sigma, 1e-9)

    # Calculate improvement
    imp = best_y - mu - xi

    # Calculate expected improvement
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)

    # Set EI to 0 where sigma is very small
    ei[sigma < 1e-9] = 0.0

    return ei


def find_next_sample_point(gp, param_bounds, param_names, best_y, n_restarts=10):
    """
    Find the next sample point by maximizing expected improvement.

    Parameters:
    -----------
    gp : GaussianProcessRegressor
        Fitted GP model
    param_bounds : list of tuples
        List of (min, max) bounds for each parameter
    param_names : list
        List of parameter names
    best_y : float
        Current best function value
    n_restarts : int
        Number of restarts for optimization

    Returns:
    --------
    next_x : dict
        Dictionary with parameter values for the next point
    """
    from scipy.optimize import minimize

    n_params = len(param_bounds)

    # Generate multiple random starting points
    x_starts = latin_hypercube_design(n_restarts, param_bounds)

    best_x = None
    best_ei = -1

    # Run optimization from each starting point
    for x_start in x_starts:
        # Define bounds for the optimizer
        bounds = param_bounds

        # Objective function to minimize (negative EI)
        def objective(x):
            return -expected_improvement(gp, x.reshape(1, -1), best_y)[0]

        # Minimize negative EI
        result = minimize(
            objective,
            x_start,
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success and -result.fun > best_ei:
            best_ei = -result.fun
            best_x = result.x

    # If optimization failed, use the best point from the starts
    if best_x is None:
        best_idx = np.argmax(expected_improvement(gp, x_starts, best_y))
        best_x = x_starts[best_idx]

    # Convert to dictionary
    next_point = {}
    for i, param in enumerate(param_names):
        next_point[param] = best_x[i]

    return next_point


def sequential_optimization(important_params, param_levels, target_params, voltages,
                            n_initial=10, n_iterations=20, tol=0.01):
    """
    Perform sequential optimization using GP and EI.

    Parameters:
    -----------
    important_params : list
        List of important parameter names
    param_levels : dict
        Dictionary with (min, max) values for each parameter
    target_params : dict
        Target parameter values to match
    voltages : numpy array
        Voltage values for evaluation
    n_initial : int
        Number of initial design points
    n_iterations : int
        Maximum number of iterations
    tol : float
        Convergence tolerance

    Returns:
    --------
    results : dict
        Dictionary with optimization results
    """
    print(f"\nStarting sequential optimization with {len(important_params)} parameters")
    print(f"Initial points: {n_initial}, Max iterations: {n_iterations}")

    # Create parameter bounds
    param_bounds = [param_levels[p] for p in important_params]

    # Generate initial design
    X_design = latin_hypercube_design(n_initial, param_bounds)

    # Evaluate initial design points
    X_list = []
    y_list = []

    print(f"Evaluating {n_initial} initial points...")

    # Start with base parameter set (e.g., WT)
    base_params = get_parameters_WT()

    for i in range(n_initial):
        # Create parameter set for this point
        point_params = base_params.copy()
        for j, param in enumerate(important_params):
            point_params[param] = X_design[i, j]

        # Update dependent parameters
        if 'x14' in important_params and 'x14' in point_params:
            point_params['x16'] = point_params['x14']
        if 'x18' in important_params and 'x18' in point_params:
            point_params['x20'] = point_params['x18']
        if 'x19' in important_params and 'x19' in point_params:
            point_params['x21'] = point_params['x19']

        # Compute discrepancy
        try:
            disc = compute_model_discrepancy(point_params, target_params, voltages)
            X_list.append([X_design[i, j] for j in range(len(important_params))])
            y_list.append(disc)
            print(f"  Point {i + 1}/{n_initial}: Discrepancy = {disc:.4f}")
        except Exception as e:
            print(f"  Error in point {i + 1}: {e}")
            # Skip this point

    X = np.array(X_list)
    y = np.array(y_list)

    # History for tracking progress
    history = {
        'X': X.tolist(),
        'y': y.tolist(),
        'params': important_params,
        'iterations': []
    }

    # Main optimization loop
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")

        # Find current best point
        best_idx = np.argmin(y)
        best_y_val = y[best_idx]

        print(f"Current best discrepancy: {best_y_val:.4f}")

        # Exit if tolerance reached
        if best_y_val < tol:
            print(f"Convergence reached (discrepancy < {tol})")
            break

        # Fit GP model
        gp = update_gaussian_process(X, y)

        # Find next point
        next_point = find_next_sample_point(gp, param_bounds, important_params, best_y_val)

        # Evaluate next point
        point_params = base_params.copy()
        for param, value in next_point.items():
            point_params[param] = value

        # Update dependent parameters
        if 'x14' in important_params and 'x14' in point_params:
            point_params['x16'] = point_params['x14']
        if 'x18' in important_params and 'x18' in point_params:
            point_params['x20'] = point_params['x18']
        if 'x19' in important_params and 'x19' in point_params:
            point_params['x21'] = point_params['x19']

        try:
            disc = compute_model_discrepancy(point_params, target_params, voltages)

            next_x = [next_point[p] for p in important_params]
            X = np.vstack([X, next_x])
            y = np.append(y, disc)

            print(f"New point discrepancy: {disc:.4f}")

            # Update history
            history['X'].append(next_x)
            history['y'].append(disc)
            history['iterations'].append({
                'iteration': iteration + 1,
                'point': next_point,
                'discrepancy': disc
            })
        except Exception as e:
            print(f"Error evaluating next point: {e}")
            # If evaluation fails, add a point with high discrepancy
            # This will discourage the algorithm from exploring this region
            X = np.vstack([X, [next_point[p] for p in important_params]])
            y = np.append(y, 10.0 * np.max(y))  # Use a high discrepancy value

    # Get the best parameter set
    best_idx = np.argmin(y)
    best_x = X[best_idx]

    optimal_params = base_params.copy()
    for i, param in enumerate(important_params):
        optimal_params[param] = best_x[i]

    # Update dependent parameters
    if 'x14' in important_params:
        optimal_params['x16'] = optimal_params['x14']
    if 'x18' in important_params:
        optimal_params['x20'] = optimal_params['x18']
    if 'x19' in important_params:
        optimal_params['x21'] = optimal_params['x19']

    print(f"\nOptimization complete. Final best discrepancy: {y[best_idx]:.4f}")

    return {
        'optimal_params': optimal_params,
        'history': history,
        'best_discrepancy': y[best_idx]
    }


def plot_optimization_history(history):
    """
    Plot the optimization history.

    Parameters:
    -----------
    history : dict
        Optimization history

    Returns:
    --------
    fig : matplotlib figure
        The generated figure
    """
    y_values = history['y']
    n_points = len(y_values)
    n_initial = n_points - len(history['iterations'])

    # Compute best so far
    best_so_far = [min(y_values[:i + 1]) for i in range(n_points)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all evaluations
    ax.plot(range(1, n_points + 1), y_values, 'o-', label='All evaluations', alpha=0.7)

    # Plot best so far
    ax.plot(range(1, n_points + 1), best_so_far, 'r-', linewidth=2, label='Best so far')

    # Mark initial design boundary
    ax.axvline(n_initial + 0.5, color='k', linestyle='--',
               label=f'Initial design ({n_initial} points)')

    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Discrepancy')
    ax.set_title('Optimization History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Use log scale for y-axis if range is large
    if max(y_values) / (min(y_values) + 1e-10) > 10:
        ax.set_yscale('log')

    return fig


# -----------------------------
# 3. Evaluation and Validation
# -----------------------------

def evaluate_optimized_model(optimal_params, wt_params, ko_params, voltages):
    """
    Evaluate the optimized model and compare with WT and KO.

    Parameters:
    -----------
    optimal_params : dict
        Optimized parameter values
    wt_params : dict
        Wild-type parameter values
    ko_params : dict
        Knockout parameter values
    voltages : numpy array
        Voltage values for evaluation

    Returns:
    --------
    fig : matplotlib figure
        The generated figure
    """
    # Compute SSA and SSI
    ssa_wt, ssi_wt = compute_SSA_SSI(voltages, wt_params)
    ssa_ko, ssi_ko = compute_SSA_SSI(voltages, ko_params)
    ssa_opt, ssi_opt = compute_SSA_SSI(voltages, optimal_params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot SSA
    ax1.plot(voltages, ssa_wt, 'r-^', label='WT', markersize=4)
    ax1.plot(voltages, ssa_ko, 'b-o', label='KO (Target)', markersize=4)
    ax1.plot(voltages, ssa_opt, 'g--s', label='Optimized', markersize=4)
    ax1.set_xlabel('Voltage (mV)')
    ax1.set_ylabel('SSA (Open Probability)')
    ax1.set_title('Steady-State Activation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot SSI (as availability)
    ax2.plot(voltages, 1.0 - ssi_wt, 'r-^', label='WT', markersize=4)
    ax2.plot(voltages, 1.0 - ssi_ko, 'b-o', label='KO (Target)', markersize=4)
    ax2.plot(voltages, 1.0 - ssi_opt, 'g--s', label='Optimized', markersize=4)
    ax2.set_xlabel('Voltage (mV)')
    ax2.set_ylabel('Availability (1 - SSI)')
    ax2.set_title('Steady-State Availability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


def plot_current_voltage(wt_params, ko_params, optimal_params, voltages):
    """
    Plot current-voltage relationships for WT, KO, and optimized models.

    Parameters:
    -----------
    wt_params : dict
        Wild-type parameter values
    ko_params : dict
        Knockout parameter values
    optimal_params : dict
        Optimized parameter values
    voltages : numpy array
        Voltage values for evaluation

    Returns:
    --------
    fig : matplotlib figure
        The generated figure
    """
    # Create a denser voltage range for smoother curves
    v_dense = np.linspace(min(voltages), max(voltages), 100)

    # Compute current for each model at each voltage
    i_wt = []
    i_ko = []
    i_opt = []

    for v in v_dense:
        # Get steady-state distributions
        pi_wt = get_stationary_distribution(v, wt_params)
        pi_ko = get_stationary_distribution(v, ko_params)
        pi_opt = get_stationary_distribution(v, optimal_params)

        # Compute currents (using open state probability)
        # I = g * Po * (V - E_Na)
        g_Na = 12.0  # Example conductance
        e_Na = 40.0  # Example reversal potential

        i_wt.append(g_Na * pi_wt[8] * (v - e_Na))
        i_ko.append(g_Na * pi_ko[8] * (v - e_Na))
        i_opt.append(g_Na * pi_opt[8] * (v - e_Na))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(v_dense, i_wt, 'r-', label='WT', linewidth=2)
    ax.plot(v_dense, i_ko, 'b--', label='KO', linewidth=2)
    ax.plot(v_dense, i_opt, 'g-.', label='Optimized', linewidth=2)

    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Current Density (pA/pF)')
    ax.set_title('Current-Voltage Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def simulate_state_distributions(wt_params, ko_params, optimal_params=None):
    """
    Simulate and plot state distributions during action potential.

    Parameters:
    -----------
    wt_params : dict
        Wild-type parameter values
    ko_params : dict
        Knockout parameter values
    optimal_params : dict, optional
        Optimized parameter values

    Returns:
    --------
    fig : matplotlib figure
        The generated figure
    """
    # Simulate AP
    t, V = simulate_AP(t_max=200, dt=0.1)

    # Simulate channel dynamics
    print("Simulating WT channel dynamics...")
    P_wt = simulate_channel_dynamics(t, V, wt_params)

    print("Simulating KO channel dynamics...")
    P_ko = simulate_channel_dynamics(t, V, ko_params)

    P_opt = None
    if optimal_params is not None:
        print("Simulating Optimized channel dynamics...")
        P_opt = simulate_channel_dynamics(t, V, optimal_params)

    # Select time points of interest (matching Figure 13 in the paper)
    time_points = [11, 20, 50, 100, 150]

    # Plot distributions
    if optimal_params is None:
        return plot_state_distributions(t, V, P_wt, P_ko, time_points)
    else:
        # This would need a modified version of plot_state_distributions to include optimal
        # For now, we'll use the existing function
        return plot_state_distributions(t, V, P_wt, P_ko, time_points)


# -----------------------
# 4. Demo Functions
# -----------------------

def run_parameter_screening_demo():
    """
    Demonstrate parameter screening to identify important variables.
    """
    print("\n=== Running Parameter Screening Demo ===")

    # Get parameter levels and nominal values
    param_dict = get_parameters_WT()

    # Define parameter ranges (±20% of nominal)
    param_levels = {}
    for param, value in param_dict.items():
        # Skip derived parameters
        if param in ['x16', 'x20', 'x21']:
            continue

        # Fixed parameters
        if param in ['x7', 'x8', 'x9', 'x10']:
            param_levels[param] = (value, value)
            continue

        # Define range
        factor = 0.2
        param_levels[param] = (value * (1 - factor), value * (1 + factor))

    # Run screening for a reduced set of parameters (for demonstration)
    # In practice, you would screen all parameters
    demo_params = sorted([p for p in param_levels.keys()
                          if p not in ['x7', 'x8', 'x9', 'x10']])[:10]
    demo_levels = {p: param_levels[p] for p in demo_params}

    # Screen parameters
    voltages = np.arange(-140, 21, 20)  # Coarse grid for speed
    results = screen_parameters(param_dict, demo_levels, voltages, fraction=3)

    # Display plots
    results['ssa_fig'].suptitle('SSA Parameter Effects (Demo)')
    results['ssi_fig'].suptitle('SSI Parameter Effects (Demo)')

    plt.figure(results['ssa_fig'].number)
    plt.show()

    plt.figure(results['ssi_fig'].number)
    plt.show()

    return results


def run_optimization_demo():
    """
    Demonstrate optimization to find parameters that transform WT to KO.
    """
    print("\n=== Running Optimization Demo ===")

    # Get parameter sets
    p_WT = get_parameters_WT()
    p_KO = get_parameters_KO()

    # Define a small set of important parameters (for demonstration)
    # In practice, these would come from screening
    important_params = ['x1', 'x2', 'x5', 'x6', 'x17']

    # Define parameter ranges
    param_levels = {}
    for param in important_params:
        wt_val = p_WT[param]
        ko_val = p_KO[param]

        # Set range to include both WT and KO values, plus some margin
        low = min(wt_val, ko_val) * 0.8
        high = max(wt_val, ko_val) * 1.2
        param_levels[param] = (low, high)

    # Run optimization (reduced iterations for demo)
    voltages = np.arange(-140, 21, 20)  # Coarse grid for speed
    results = sequential_optimization(
        important_params,
        param_levels,
        p_KO,  # Target is KO
        voltages,
        n_initial=5,
        n_iterations=10,
        tol=0.05
    )

    # Plot optimization history
    history_fig = plot_optimization_history(results['history'])
    plt.figure(history_fig.number)
    plt.show()

    # Evaluate optimized model
    voltages_fine = np.arange(-140, 21, 5)  # Finer grid for evaluation
    eval_fig = evaluate_optimized_model(
        results['optimal_params'],
        p_WT,
        p_KO,
        voltages_fine
    )
    plt.figure(eval_fig.number)
    plt.show()

    return results


def run_full_demo():
    """
    Run a full demonstration of the workflow.
    """
    print("\n=== Starting Full Demonstration ===")

    # Step 1: Obtain parameter sets
    p_WT = get_parameters_WT()
    p_KO = get_parameters_KO()

    # Step 2: Compare basic model behavior
    voltages_fine = np.arange(-140, 21, 2)

    print("\n--- Comparing Basic Model Behavior ---")
    ssa_wt, ssi_wt = compute_SSA_SSI(voltages_fine, p_WT)
    ssa_ko, ssi_ko = compute_SSA_SSI(voltages_fine, p_KO)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(voltages_fine, ssa_wt, 'r-^', label='WT: SSA', markersize=4)
    ax1.plot(voltages_fine, 1.0 - ssi_wt, 'b-^', label='WT: Availability', markersize=4)
    ax1.plot(voltages_fine, ssa_ko, 'r--o', label='KO: SSA', markersize=4)
    ax1.plot(voltages_fine, 1.0 - ssi_ko, 'b--o', label='KO: Availability', markersize=4)
    ax1.set_xlabel('Voltage (mV)')
    ax1.set_ylabel('Steady-State Probability')
    ax1.set_title('Steady-State Activation & Availability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    plt.show()

    # Step 3: Run parameter screening (limited scope for demo)
    print("\n--- Running Parameter Screening (Limited Scope for Demo) ---")

    # Define parameter ranges (±20% of nominal)
    param_levels = {}
    for param, value in p_WT.items():
        # Skip derived parameters
        if param in ['x16', 'x20', 'x21']:
            continue

        # Fixed parameters
        if param in ['x7', 'x8', 'x9', 'x10']:
            param_levels[param] = (value, value)
            continue

        # Define range
        factor = 0.2
        param_levels[param] = (value * (1 - factor), value * (1 + factor))

    # For demo, use a smaller set of parameters
    demo_params = sorted([p for p in param_levels.keys()
                          if p not in ['x7', 'x8', 'x9', 'x10']])[:10]
    demo_levels = {p: param_levels[p] for p in demo_params}

    # Screen parameters
    voltages_coarse = np.arange(-140, 21, 20)  # Coarse grid for speed
    screen_results = screen_parameters(p_WT, demo_levels, voltages_coarse, fraction=3)

    # Display plots
    plt.figure(screen_results['ssa_fig'].number)
    plt.show()

    plt.figure(screen_results['ssi_fig'].number)
    plt.show()

    # Step 4: Run optimization (using parameters identified by screening)
    print("\n--- Running Optimization (with Screened Parameters) ---")

    # Use results from screening (in a real run, but for demo we'll define manually)
    important_params = screen_results['ssa_important'][:5]  # Take top 5 for demo

    # Run optimization
    opt_results = sequential_optimization(
        important_params,
        {p: param_levels[p] for p in important_params},
        p_KO,  # Target is KO
        voltages_coarse,
        n_initial=5,
        n_iterations=10,
        tol=0.05
    )

    # Plot optimization history
    history_fig = plot_optimization_history(opt_results['history'])
    plt.figure(history_fig.number)
    plt.show()

    # Step 5: Evaluate optimized model
    print("\n--- Evaluating Optimized Model ---")

    eval_fig = evaluate_optimized_model(
        opt_results['optimal_params'],
        p_WT,
        p_KO,
        voltages_fine
    )
    plt.figure(eval_fig.number)
    plt.show()

    # Step 6: Compare current-voltage relationships
    iv_fig = plot_current_voltage(p_WT, p_KO, opt_results['optimal_params'], voltages_fine)
    plt.figure(iv_fig.number)
    plt.show()

    # Step 7: Analyze state distributions
    print("\n--- Analyzing State Distributions ---")
    state_fig = simulate_state_distributions(p_WT, p_KO, opt_results['optimal_params'])
    plt.figure(state_fig.number)
    plt.show()

    print("\n=== Demonstration Complete ===")

    return {
        'screen_results': screen_results,
        'opt_results': opt_results,
        'p_WT': p_WT,
        'p_KO': p_KO,
        'p_opt': opt_results['optimal_params']
    }

# Example usage:
# 1. Run parameter screening demo
results = run_parameter_screening_demo()

# 2. Run optimization demo
# opt_results = run_optimization_demo()

# 3. Run full demo
# all_results = run_full_demo()