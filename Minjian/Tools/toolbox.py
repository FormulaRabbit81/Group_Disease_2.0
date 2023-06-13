"""This module contains various functions for Hawkes Processes and visualization.

Functions:
- convert_date_to_numberB: Convert dates to numbers.
- distribute: Distribute cases evenly within the same day.
- binary_search: Perform binary search to find the right position for insertion.
- _history_n: Reduce computations by considering event times within a certain number of days.
- intensity_lambda_dep: Compute the intensity function with depth.
- _integral_term: Calculate the integral term of the log-likelihood function.
- _log_term: Calculate the logarithmic term of the log-likelihood function.
- _llh_neg: Calculate the negative log-likelihood function.
- MLE: Minimize the negative log-likelihood function.
- hawkes_prediction: Predict future events using the Hawkes process.
- prediction: Generate event predictions using the Hawkes process.
- QQ_plot: Generate a Q-Q plot to compare observed and theoretical quantiles.
- KS_plot: Plot the Kolmogorov-Smirnov (KS) plot.

Usage Example:
-------------
# Importing the module
from your_module_name import *

# Converting a date to a number
date = "15/06/2022"
number = convert_date_to_numberB(date)

# Distributing event times
data = [10, 11, 11, 12, 12, 12]
distributed_data = distribute(data)

# Calculating the intensity function
intensity = intensity_lambda_dep(5.0, event_times, depth=30, kernel=kernel_func, base=base_func)

# Maximize the log-likelihood
result = MLE(intensity_func, event_times, initial_params)

# Generating event predictions
prediction(intensity_func, history_events, T_i, p=0.5, n=50, color="b", alpha=0.5)

# Creating a Q-Q plot
QQ_plot(observed, theoretical)

# Creating a KS plot
KS_plot(samples, event_times, color="r", alpha=0.2)
"""


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import scipy.integrate as spi
import scipy.optimize as opt


def convert_date_to_numberB(date, earliest_date="31/12/2010"):
    """convert dates to numbers

    Args:
        date (str): String of a date, it should be in format "%d/%m/%Y" 
        earliest_date (str, optional): Date that has value 0. Defaults to "01/01/2011".

    Returns:
        int: Number of days after the earliest_date
    """
    date_format = "%d/%m/%Y"
    delta = datetime.strptime(date, date_format) - datetime.strptime(earliest_date, date_format)
    return delta.days


def convert_date_to_numberC(date, earliest_date="2013-12-25"):
    """convert dates to numbers

    Args:
        date (str): String of a date, it should be in format "%d/%m/%Y" 
        earliest_date (str, optional): Date that has value 0. Defaults to "01/01/2011".

    Returns:
        int: Number of days after the earliest_date
    """
    date_format = "%Y-%m-%d" 
    delta = datetime.strptime(date, date_format) - datetime.strptime(earliest_date, date_format)
    return delta.days


def distribute(data):
    """Distribute cases in the same day evenly

    Args:
        data (List(int)): A list of event times

    Returns:
        List: Distributed event times
    """
    counter = Counter(data)
    sorted_elements = sorted(counter.keys())
    time_ticks = []
    for e in sorted_elements:
        count = counter[e]
        for n in range(count):
            time_ticks.append(int(e)-1+(n+1)/count)

    return time_ticks


def binary_search(sorted_list, new):
    """Find the right position for a insertion"""
    left = 0
    right = len(sorted_list) - 1

    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] == new:
            return mid + 1
        elif sorted_list[mid] < new:
            left = mid + 1
        else:
            right = mid - 1

    return left

def intensity_lambda_dep(t, event_times, depth=None, kernel=None, base=None):
    """The intensity function (lambda) with depth.

    Args:
        t (float): Time
        event_times (array-like): Event times
        depth (int, optional): Number of days to consider. Defaults to 30.
        **kwargs: Additional keyword arguments for phi and lambda0

    Returns:
        float: Intensity value
    """
    self_exciting_term = 0
    n = binary_search(event_times, t)
    if depth:
        for ind in range(n - 1, -1, -1):
            if (t - event_times[ind]) >= depth: 
                return base(t) + self_exciting_term
            else:
                self_exciting_term += kernel(t - event_times[ind])
    else:
        for ind in range(n):
            self_exciting_term += kernel(t - event_times[ind])

    return base(t) + self_exciting_term

def _integral_term(paras, intensity_func, event_times, model=None):
    """Calculate the integral term of the log-likelihood function.

    Args:
        x (tuple): Parameters for the intensity function.
        intensity_func (Function): Intensity function.
        time_ticks_m_small (list): List of time ticks.

    Returns:
        float: Integral term.

    """
    if model == "constexp":
        alpha, delta, mu = paras
        m = alpha/delta
        T = event_times[-1]
        n = len(event_times)
        integral = mu * T + (n - 1) * m
        for ind in range(n):
            integral -= np.exp(-delta * (T-event_times[ind]))
        return integral
    
    if model == "HN":
        alpha, delta, mu, N = paras
        m = alpha/delta
        t = event_times
        T = t[-1]
        n = len(event_times)
        integral = 0
        for ind in range(n-1):
            for l in range(ind, n-1, 1):
                integral += (1 - l / N) * (np.exp(-delta * (t[l]-t[ind]))-np.exp(-delta*(t[l+1]-t[ind])))
        return m * integral

    elif model == "SIR":
        mu, N = paras
        t = event_times
        T = t[-1]
        integral = 0
        for ind in range(len(t)-1):
            integral -= ind * (t[ind+1] - t[ind-1])
        return integral * mu / N + mu * T

    else:
        integral = 0
        for ind in range(len(event_times) - 1):
            i = spi.quad(lambda t: intensity_func(t, paras), event_times[ind], event_times[ind+1], epsabs=1e-15)
            integral += i[0]
        return integral

def _log_term(paras, intensity_func, event_times, model=None, depth=None):
    """Calculate the logarithmic term of the log-likelihood function.

    Args:
        x (tuple): Parameters for the intensity function.
        intensity_func (function): Intensity function.
        time_ticks_m_small (list): List of time ticks.

    Returns:
        float: Logarithmic term.
    """
    if model == "constexp":
        alpha, delta, mu = paras
        t = event_times
        n = len(t)
        exp = 0
        res = 0
        if depth:
            for ind in range(n - 1, -1, -1):
                if t[-1] - t[ind]:
                    return res
                else:
                    res += np.log(mu + exp)
                    exp = np.exp(-delta*(t[ind]-t[ind-1])) * (exp + alpha)
            return res 
        else:
            for ind in range(1, n, 1):
                res += np.log(mu + exp)
                exp = np.exp(-delta*(t[ind]-t[ind-1])) * (exp + alpha)
            return res 
    
    elif model == "HN":
        alpha, delta, mu, N = paras
        t = event_times
        n = len(t)
        exp = 0
        res = 0
        for ind in range(1, n, 1):
            res += (np.log(mu + exp) - np.log(1 - (ind-1)/N) + np.log(1 - ind/N))
            exp = np.exp(-delta*(t[ind]-t[ind-1])) * (exp + alpha)
        return res 
    
    elif model == "SIR":
        mu, N = paras
        res = 0
        for ind in range(len(event_times)):
            res += np.log(mu - ind * mu / N)
        return res
        
    else:
        res = 0
        for t in event_times:
            res += np.log(intensity_func(t, paras))
        return res

def _llh_neg(paras, intensity_func, event_times, model=None):
    """Calculate the negative log-likelihood function so that minimize maximizes the llh.

    Args:
        x (tuple): Parameters for the intensity function.
        **kwargs: Additional keyword arguments.
            intensity_func (function): Intensity function.
            time_ticks_m_small (list): List of time ticks.

    Returns:
        float: Negative log-likelihood.

    """
    return _integral_term(paras, intensity_func, event_times, model=model) - _log_term(paras, intensity_func, event_times, model=model)

def MLE(intensity_func, event_times, initial_params, model=None , **kwargs):
    """Minimize the negative log-likelihood function.

    Args:
        intensity_func (Function): Intensity function.
        event_times (list): List of event times.
        initial_params (tuple): Initial parameters for the intensity function.
        **kwargs: Additional keyword arguments to be passed to the log-likelihood function.

    Returns:
        OptimizeResult: The optimization result.

    """
    
    result = opt.minimize(lambda x: _llh_neg(x, intensity_func, event_times, model=model),
                          initial_params, **kwargs)
    return result

def MLE_de(intensity_func, event_times, model=None, **kwargs):
    """Minimize the negative log-likelihood function.

    Args:
        intensity_func (Function): Intensity function.
        event_times (list): List of event times.
        initial_params (tuple): Initial parameters for the intensity function.
        **kwargs: Additional keyword arguments to be passed to the log-likelihood function.

    Returns:
        OptimizeResult: The optimization result.

    """
    
    result = opt.differential_evolution(lambda x: _llh_neg(x, intensity_func, event_times, model=model), **kwargs)
    return result

def simulate_hawkes_process(intensity_func, history_events, T): 
    """ Only works for monotonically decreasing kernels
    """

    simulated_events = history_events.copy()
    
    if history_events:
        T_i = history_events[-1]
    else:
        T_i = 0
    
    while T_i < T:
        lambda_star = intensity_func(T_i+0.000000001, simulated_events)
        u = np.random.uniform(0, 1)
        tau = -np.log(u) / lambda_star
        T_i += tau
        s = np.random.uniform(0, 1)

        if s <= (intensity_func(T_i+0.000000001, simulated_events)/ lambda_star):
            simulated_events.append(T_i)
    return simulated_events

def hawkes_prediction(intensity_func, history_events, p=0.2):
    """Use Hawkes Processes to predict time of future events by thinning algorithm

    Args:
        intensity_func (func): the intensity function 
        p (float): proportion of predictions to history events
        history_events (Iterable(floats)): history events for the prediction

    Returns:
        List(float): prediction 
    """
    T = history_events[-1] * (1 + p)
    return simulate_hawkes_process(intensity_func, history_events, T)

def prediction(intensity_func, history_events, T_i, p=0.5, n=50, **kwargs):
    """Generate event predictions using the Hawkes process.

    Args:
        intensity_func (function): Intensity function.
        history_events (list): List of historical event times.
        T_i (float): Time interval.
        p (float, optional): Extra time to predict beyond T_i as a fraction of T_i. Defaults to 0.5.
        n (int, optional): Number of predictions to generate. Defaults to 50.
        **kwargs: Additional keyword arguments for customizing the plot, including color and alpha.

    Returns:
        None

    Raises:
        ValueError: If T_i is not a positive number.
        ValueError: If p is not in the range of [0, 1].

    """
    if T_i <= 0:
        raise ValueError("T_i must be a positive number.")
    if not 0 <= p <= 1:
        raise ValueError("p must be in the range of [0, 1].")

    T = T_i * (1 + p)
    N1 = binary_search(history_events, T)
    N0 = binary_search(history_events, T_i)

    color = kwargs.pop("color", None) if kwargs is not None else "k"
    alpha = kwargs.pop("alpha", None) if kwargs is not None else 0.15

    for _ in range(n):
        pred = hawkes_prediction(intensity_func, history_events, p)
        N = len(pred)
        plt.plot(pred, np.arange(N + N0)[-N:], color=color, alpha=alpha, **kwargs)

    plt.plot(history_events[:N1], range(N1), label="data", color="r")
    plt.xlabel("Time")
    plt.ylabel("N(t)")
    plt.legend()

    plt.show()
    
def simulate_cluster_structure(kernel, base, T):
    rate = base.c
    T_i = 0
    G = []
    while T_i < T:
        T_i += np.random.exponential(1/rate)
        G.append(T_i)
    event_times = G.copy()
    m = kernel.m()
    while len(G) > 0:
        inter_arrival_times = np.array([])
        for t in G:
            C = np.random.poisson(m)
            inter_arrival_times = np.append(inter_arrival_times, np.random.exponential(m, C) + t)
        G = [t for t in inter_arrival_times if t < T]
        event_times += G
    return sorted(list(set(event_times)))

def QQ_plot(observed, theoretical):
    """Generate a Q-Q plot to compare observed quantiles with theoretical quantiles.

    Args:
        observed (Iterable): Array of observed quantiles.
        theoretical (Iterable): Array of theoretical quantiles.

    Returns:
        None

    """
    plt.scatter(theoretical, observed, marker=".", color="k")
    plt.plot([np.min(theoretical), np.max(theoretical)],
            [np.min(observed), np.max(observed)],
            color='red')

    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Observed Quantiles')
    plt.title('Q-Q Plot')

    plt.show()
    
def KS_plot(samples, event_times, **kwargs):
    """Plot the Kolmogorov-Smirnov (KS) plot.

    Args:
        samples (list): List of samples.
        event_times (array-like): Event times or data.
        **kwargs: Additional keyword arguments for customization.
            - color (str): Color of the plotted lines (default: 'k').
            - alpha (float): Transparency of the plotted lines (default: 0.15).

    Returns:
        None
    """
    
    color = kwargs.get("color", "k") if kwargs is not None else "k"
    alpha = kwargs.get("alpha", 0.15) if kwargs is not None else 0.15
    N = len(samples)
    for sample in samples:
        n = len(sample)
        plt.plot(sample, np.arange(1, n + 1) / n, color=color, alpha=alpha)

    m = len(event_times)
    plt.plot(event_times, np.arange(1, m + 1) / m, label="data", color="r")
    plt.xlabel('Quantiles')
    plt.ylabel('Cumulative Probability')
    plt.title(f'KS Plot from {N} samples')
    plt.show()

def intensity_constructor(t, paras, event_times, kernel, base, SIR=False, depth=None):
    kn = kernel().n
    bn = base().n
    kernel_func = kernel(*paras[:kn])
    base_func = base(*paras[kn:(kn + bn)])
    if SIR:
        N = paras[-1]
        return (1 - len(event_times) / N) * intensity_lambda_dep(t, event_times, depth=depth, kernel=kernel_func, base=base_func)
    else:
        return intensity_lambda_dep(t, event_times, depth=depth, kernel=kernel_func, base=base_func)

def bounds(kernel, base):
    return kernel().bounds + base().bounds

def constraints(kernel, base):
    pass

def newton_raphson_method(f, grad, hess, x0, bounds=None, max_iter=100, tol=1e-6):
    x = x0
    converged = False

    for _ in range(max_iter):
        # Evaluate the function and its gradient
        f_val = f(x)
        g = grad(x)

        # Check if the solution satisfies the tolerance
        if np.abs(f_val) < tol:
            converged = True
            break

        # Compute the Hessian matrix
        H = hess(x)

        # Solve the Newton-Raphson update equation
        delta_x = np.linalg.solve(H, -g)

        # Update the solution
        x_new = x + delta_x

        # Apply bounds to the updated solution
        if bounds is not None:
            x_new = np.clip(x_new, bounds[0], bounds[1])

        # Check if the solution has converged
        if np.allclose(x, x_new, rtol=tol, atol=tol):
            converged = True
            break

        # Update the solution for the next iteration
        x = x_new

    return x, converged





