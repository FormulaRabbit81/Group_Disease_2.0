"""This module provides functions for analyzing and simulating Hawkes processes, 
as well as performing maximum likelihood estimation (MLE) to estimate the parameters of the intensity function.

Functions:
- convert_date_to_numberB: Converts dates to numbers.
- convert_date_to_numberC: Converts dates to numbers.
- distribute: Distributes cases in the same day evenly.
- binary_search: Finds the right position for insertion.
- intensity_lambda_dep: Calculates the intensity function with depth.
- _integral_term: Calculates the integral term of the log-likelihood function.
- _log_term: Calculates the logarithmic term of the log-likelihood function.
- _llh_neg: Calculates the negative log-likelihood function.
- MLE: Performs maximum likelihood estimation to find optimal parameters.
- MLE_de: Performs differential evolution to find optimal parameters.
- simulate_hawkes_process: Simulates a Hawkes process.
- simulate_cluster_structure: Simulates a cluster structure.
- hawkes_prediction: Uses Hawkes processes to predict future events.
- prediction: Generates event predictions using the Hawkes process.
- QQ_plot: Generates a Q-Q plot to compare observed and theoretical quantiles.
- KS_plot: Plots the Kolmogorov-Smirnov (KS) plot.
- intensity_constructor: Constructs the intensity function for optimization.
- bounds: Gets the bounds for the parameters.

Note: This module requires the numpy, datetime, matplotlib.pyplot, collections, scipy.integrate, and scipy.optimize libraries.

"""


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import scipy.integrate as spi
import scipy.optimize as opt


def convert_date_to_numberB(date, earliest_date="31/12/2013"):
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
            time_ticks.append(int(e)+(n+1)/count)

    return time_ticks

def binary_search(sorted_list, new):
    """Find the right position for insertion.

    Args:
        sorted_list (List): A sorted list.
        new: The element to be inserted.

    Returns:
        int: The index where the element should be inserted.
    """
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
    """Compute the intensity function lambda(t) with depth.

    Args:
        t (float): Time.
        event_times (array-like): Event times.
        depth (int, optional): Number of days to consider. Defaults to None.
        kernel (function, optional): Kernel function. Defaults to None.
        base (function, optional): Base function. Defaults to None.

    Returns:
        float: Intensity value.
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
        paras (tuple): Parameters for the intensity function.
        intensity_func (function): Intensity function.
        event_times (list): List of event times.
        model (str, optional): Model name. Defaults to None.

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

    elif model == "constpow":
        alpha, delta, eta, mu = paras
        t = event_times
        T = t[-1]
        n = len(t)
        integral = 0
        for ind in range(1, n, 1):
            integral -= (T - t[ind] + delta) ** (-eta)
        return (alpha/eta)*integral+alpha/(eta * delta**eta)+mu*T
    
    elif model == "HN":
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
        paras (tuple): Parameters for the intensity function.
        intensity_func (function): Intensity function.
        event_times (list): List of event times.
        model (str, optional): Model name. Defaults to None.
        depth (int, optional): Number of days to consider. Defaults to None.

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
    
    elif model == "constpow":
        alpha, delta, eta, mu = paras
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
    """Calculate the negative log-likelihood function.

    Args:
        paras (tuple): Parameters for the intensity function.
        intensity_func (function): Intensity function.
        event_times (list): List of event times.
        model (str, optional): Model name. Defaults to None.

    Returns:
        float: Negative log-likelihood.
    """
    return _integral_term(paras, intensity_func, event_times, model=model) - _log_term(paras, intensity_func, event_times, model=model)

def MLE(intensity_func, event_times, initial_params, model=None , **kwargs):
    """Perform maximum likelihood estimation (MLE) to find the optimal parameters.

    Args:
        intensity_func (function): Intensity function.
        event_times (list): List of event times.
        initial_params (tuple): Initial parameters for the intensity function.
        model (str, optional): Model name. Defaults to None.
        **kwargs: Additional keyword arguments for the optimization.

    Returns:
        OptimizeResult: The optimization result.
    """
    result = opt.minimize(lambda x: _llh_neg(x, intensity_func, event_times, model=model),
                          initial_params, **kwargs)
    return result

def MLE_de(intensity_func, event_times, model=None, **kwargs):
    """Perform differential evolution (DE) to find the optimal parameters.

    Args:
        intensity_func (function): Intensity function.
        event_times (list): List of event times.
        model (str, optional): Model name. Defaults to None.
        **kwargs: Additional keyword arguments for the optimization.

    Returns:
        OptimizeResult: The optimization result.
    """
    result = opt.differential_evolution(lambda x: _llh_neg(x, intensity_func, event_times, model=model), **kwargs)
    return result

def simulate_hawkes_process(intensity_func, history_events, T): 
    """Simulate a Hawkes process, only works for monotonically decreasing kernels.

    Args:
        intensity_func (function): Intensity function.
        history_events (list): List of historical event times.
        T (float): Time duration to simulate.

    Returns:
        List: Simulated event times.
    """
    simulated_events = history_events.copy()
    
    if history_events:
        T_i = history_events[-1]
    else:
        T_i = 0
    
    while T_i < T:
        lambda_star = intensity_func(T_i+1e-10, simulated_events)
        u = np.random.uniform(0, 1)
        tau = -np.log(u) / lambda_star
        T_i += tau
        s = np.random.uniform(0, 1)

        if s <= (intensity_func(T_i+1e-10, simulated_events)/ lambda_star):
            simulated_events.append(T_i)
    return simulated_events[:-1]

def simulate_cluster_structure(kernel, base, history_event, T):
    """Simulate a cluster structure with branching algorithm.

    Args:
        kernel (function): Kernel function.
        base (function): Base function.
        history_event (list): List of historical event times.
        T (float): Time duration to simulate.

    Returns:
        List: Simulated event times.
    """
    rate = base.c
    
    if history_event:
        T_i = history_event[-1]
        T_0 = history_event[-1]
    else:
        T_i = 0
        T_0 = 0
    
    event_times = history_event.copy()
    while T_i < T:
        T_i += np.random.exponential(1/rate)
        event_times.append(T_i)
    G = event_times.copy()
    m = kernel.m()
    while len(G) > 0:
        inter_arrival_times = np.array([])
        for t in G:
            C = np.random.poisson(m)
            inter_arrival_times = np.append(inter_arrival_times, np.random.exponential(m, C) + t)
        G = [t for t in inter_arrival_times if (t <= T and t >= T_0)]
        event_times += G
    return sorted(list(set(event_times)))[:-1]

def hawkes_prediction(intensity_func, history_events, p=0.2):
    """Use Hawkes Processes to predict time of future events by thinning algorithm.

    Args:
        intensity_func (function): Intensity function.
        history_events (iterable): Historical event times for the prediction.
        p (float): Proportion of predictions to history events.

    Returns:
        List: Predicted event times.
    """
    T = history_events[-1] * (1 + p)
    n = len(history_events)
    return simulate_hawkes_process(intensity_func, history_events, T)[n:]

def prediction(intensity_func, history_events, p=0.5, n=50, **kwargs):
    """Generate event predictions using the Hawkes process.

    Args:
        intensity_func (function): Intensity function.
        history_events (list): List of historical event times.
        p (float, optional): Extra time to predict beyond T_i as a fraction of T_i. Defaults to 0.5.
        n (int, optional): Number of predictions to generate. Defaults to 50.
        **kwargs: Additional keyword arguments for customizing the plot, including color and alpha.

    Returns:
        None
    """
    T_i = history_events[-1]
    T = T_i * (1 + p)
    N0 = binary_search(history_events, T_i)

    color = kwargs.pop("color", None) if kwargs is not None else "k"
    alpha = kwargs.pop("alpha", None) if kwargs is not None else 0.15

    for _ in range(n):
        pred = hawkes_prediction(intensity_func, history_events, p)
        N = len(pred)
        plt.plot(pred, np.arange(N + N0)[-N:], color=color, alpha=alpha, **kwargs)

    plt.plot(history_events, range(len(history_events)), label="data", color="r")

def QQ_plot(observed, theoretical):
    """Generate a Q-Q plot to compare observed quantiles with theoretical quantiles.

    Args:
        observed (iterable): Array of observed quantiles.
        theoretical (iterable): Array of theoretical quantiles.

    Returns:
        None
    """
    plt.scatter(theoretical, observed, marker=".", color="k")
    plt.plot([np.min(theoretical), np.max(theoretical)],
            [np.min(observed), np.max(observed)],
            color='red')
    
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
    plt.show()

def intensity_constructor(t, paras, event_times, kernel, base, SIR=False, depth=None):
    """Construct the intensity function for optimization.

    Args:
        t (float): Time.
        paras (tuple): Parameters for the intensity function.
        event_times (list): List of event times.
        kernel (class): Kernel class.
        base (class): Base class.
        SIR (bool, optional): Flag indicating if SIR model is used. Defaults to False.
        depth (int, optional): Number of days to consider. Defaults to None.

    Returns:
        float: Intensity value.
    """
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
    """Get the bounds for the parameters.

    Args:
        kernel (class): Kernel class.
        base (class): Base class.

    Returns:
        tuple: Lower and upper bounds for the parameters.
    """
    return kernel().bounds + base().bounds

def count_by_days(times, m, n=7):
    counts = []
    start = int(times[0])
    end = start + 7
    pos = 0
    for _ in range(m):
        counts.append(binary_search(times, end) - pos)
        pos += counts[-1]
        end += 7
    return np.array(counts)
