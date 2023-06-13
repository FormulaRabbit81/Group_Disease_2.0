"""This module defines various functions and kernels used in Hawkes Process.

Classes:
- Function: Base class for functions.
- Kernel: Base class for kernel functions.
- ExpKernel: Exponential kernel function.
- RayleighKernel: Rayleigh kernel function.
- PowerLawKernel: Power-law kernel function.
- Base: Base class for base functions.
- ConstBase: Constant base function.
- PeriodicBase: Periodic base function.

Usage Example:
-------------
# Importing the module
from your_module_name import *

# Creating an instance of ExpKernel
exp_kernel = ExpKernel(alpha=0.5, delta=0.2)

# Evaluating the kernel function
result = exp_kernel(2.0)

# Creating an instance of ConstBase
const_base = ConstBase(c=3.0)

# Evaluating the base function
value = const_base(1.5)
"""


import numpy as np


class Function:
    """Base class for functions."""


class Kernel(Function):
    """Base class for kernel functions."""


class ZeroKernel(Kernel):
    
    n = 0
    def __init__(self, *args):
        pass
    
    def __call__(self, _):
        return 0
    
    def m(self):
        return 0 


class ExpKernel(Kernel):

    n = 2
    bounds = ((0, np.inf), (0, np.inf))

    def __init__(self, *args):
        if len(args) == 2:
            self.alpha = args[0]
            self.delta = args[1]
    
    def __call__(self, t, *args):
        return self.alpha * np.exp(-self.delta*t)
    
    def integrate(self, t):
        return self.alpha / self.delta * (1 - np.exp(-self.delta*t)) 
    
    def m(self):
        return self.alpha / self.delta 


class RayleighKernel(Kernel):

    n = 2
    bounds = ((0, np.inf), (0, np.inf))

    def __init__(self, *args):
        if len(args) == 2:
            self.alpha = args[0]
            self.delta = args[1]
    
    def __call__(self, t, *args):
        return self.alpha * t * np.exp(-self.delta*(t**2)/2)
    
    def integrate(self, t):
        return self.alpha / self.delta * (1 - np.exp(-self.delta*t**2)/2)

    def m(self):
        return self.alpha / self.delta 

class PowerLawKernel(Kernel):

    n = 3
    bounds = ((0, np.inf), (0, np.inf), (0, np.inf))

    def __init__(self, *args):
        if len(args) == 3:
            self.alpha = args[0]
            self.delta = args[1]
            self.eta = args[2]
        self.constraints = ({'type': 'ineq', 'fun': lambda x: self.eta * self.delta**self.eta - self.alpha})
    
    def __call__(self, t, *args):
        return self.alpha / ((t + self.delta)**(self.eta + 1))

    def integrate(self, t):
        return (self.alpha / self.eta) * (self.delta**(-self.eta) - (self.delta + t)**(-self.eta))

    def m(self):
        return self.alpha / self.eta / (self.delta ** self.eta)
    
class Base(Function):
    pass


class ConstBase(Base):

    n = 1
    bounds = ((0, np.inf),)

    def __init__(self, *args):
        if len(args) == 1:
            self.c = args[0]
    
    def __call__(self, *args):
        return self.c


class PeriodicBase(Base):

    n = 4
    bounds = ((-np.inf, np.inf), (-np.inf, np.inf),
              (-np.inf, np.inf), (-np.inf, np.inf))

    def __init__(self, *args):
        if len(args) == 4:
            self.A = args[0]
            self.B = args[1]
            self.M = args[2]
            self.N = args[3]
    
    def __call__(self, t, *args):
        A = self.A
        B = self.B
        M = self.M
        N = self.N
        p = 1/365.25
        return max(A + B*t + M*np.cos(2 * np.pi * t * p)
                   + N*np.sin(2 * np.pi * t * p), 0)
