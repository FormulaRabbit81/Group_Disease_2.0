"""This module includes various classes for implementing a Hawkes Process, a self-exciting point process model.
Classes:

Function: A base class for mathematical functions.

Kernel: A base class for kernel functions used in the Hawkes Process.

ZeroKernel: A kernel function that always returns zero.
ExpKernel: An exponential kernel function.
RayleighKernel: A Rayleigh kernel function.
PowerLawKernel: A power-law kernel function.

Base: A base class for base (or background rate) functions.

ConstBase: A constant base function.
PeriodicBase: A periodic base function for modeling cyclical trends.

"""


import numpy as np


class Function:
    """Base class for functions."""


class Kernel(Function):
    """Base class for kernel functions.
    
    Kernels are used to determine the shape of the triggering function 
    in a Hawkes Process, which represents the probability of an event 
    triggering another event.
    """


class ZeroKernel(Kernel):
    """Zero kernel always returning zero regardless of input.
    
    This is effectively equivalent to a Hawkes process without self-excitation.
    """
    n = 0
    def __init__(self, *args):
        """Initializes a ZeroKernel instance."""
        pass
    
    def __call__(self, _):
        """Evaluates the ZeroKernel at a given time. Always returns zero."""
        return 0
    
    def m(self):
        """Calculates the mean (expected number of events) given the kernel. Always returns zero for ZeroKernel."""
        return 0 


class ExpKernel(Kernel):
    """Exponential kernel function.

    The Exponential kernel is a common choice for the Hawkes process, 
    due to its mathematical tractability and its property of 'memory decay'.
    """
    n = 2
    bounds = ((0, np.inf), (0, np.inf))

    def __init__(self, *args):
        """Initializes an ExpKernel with decay rate delta and scaling factor alpha."""
        if len(args) == 2:
            self.alpha = args[0]
            self.delta = args[1]
    
    def __call__(self, t, *args):
        """Evaluates the ExpKernel at a given time."""
        return self.alpha * np.exp(-self.delta*t)
    
    def integrate(self, t):
        """Calculates the integral of the kernel function from 0 to t."""
        return self.alpha / self.delta * (1 - np.exp(-self.delta*t)) 
    
    def m(self):
        """Calculates the mean (expected number of events) given the kernel."""
        return self.alpha / self.delta 


class RayleighKernel(Kernel):
    """Rayleigh kernel function."""
    n = 2
    bounds = ((0, np.inf), (0, np.inf))

    def __init__(self, *args):
        """Initializes a RayleighKernel with scaling factor alpha and shape parameter delta."""
        if len(args) == 2:
            self.alpha = args[0]
            self.delta = args[1]
    
    def __call__(self, t, *args):
        """Evaluates the RayleighKernel at a given time."""
        return self.alpha * t * np.exp(-self.delta*(t**2)/2)
    
    def integrate(self, t):
        """Calculates the integral of the kernel function from 0 to t."""
        return self.alpha / self.delta * (1 - np.exp(-self.delta*t**2)/2)

    def m(self):
        """Calculates the mean (expected number of events) given the kernel."""
        return self.alpha / self.delta 

class PowerLawKernel(Kernel):
    """Power-law kernel function."""
    n = 3
    bounds = ((0, np.inf), (0, np.inf), (0, np.inf))

    def __init__(self, *args):
        """Initializes a PowerLawKernel with scaling factor alpha, shift parameter delta and power parameter eta."""
        if len(args) == 3:
            self.alpha = args[0]
            self.delta = args[1]
            self.eta = args[2]
        self.constraints = ({'type': 'ineq', 'fun': lambda x: self.eta * self.delta**self.eta - self.alpha})
    
    def __call__(self, t, *args):
        """Evaluates the PowerLawKernel at a given time."""
        return self.alpha / ((t + self.delta)**(self.eta + 1))

    def integrate(self, t):
        """Calculates the integral of the kernel function from 0 to t."""
        return (self.alpha / self.eta) * (self.delta**(-self.eta) - (self.delta + t)**(-self.eta))

    def m(self):
        """Calculates the mean (expected number of events) given the kernel."""
        return self.alpha / self.eta / (self.delta ** self.eta)
    
class Base(Function):
    """Base class for base functions in the Hawkes process."""
    pass


class ConstBase(Base):
    """Constant base function for the Hawkes process."""
    n = 1
    bounds = ((0, np.inf),)

    def __init__(self, *args):
        """Initializes a ConstBase with constant parameter c."""
        if len(args) == 1:
            self.c = args[0]
    
    def __call__(self, *args):
        """Evaluates the ConstBase at a given time."""
        return self.c


class PeriodicBase(Base):
    """Periodic base function for modeling cyclical patterns in the Hawkes process."""
    n = 4
    bounds = ((-np.inf, np.inf), (-np.inf, np.inf),
              (-np.inf, np.inf), (-np.inf, np.inf))

    def __init__(self, *args):
        """Initializes a PeriodicBase with amplitude A, phase B, offset M, and frequency N."""
        if len(args) == 4:
            self.A = args[0]
            self.B = args[1]
            self.M = args[2]
            self.N = args[3]
    
    def __call__(self, t, *args):
        """Evaluates the PeriodicBase at a given time."""
        A = self.A
        B = self.B
        M = self.M
        N = self.N
        p = 1/365.25
        return max(A + B*t + M*np.cos(2 * np.pi * t * p)
                   + N*np.sin(2 * np.pi * t * p), 0)
