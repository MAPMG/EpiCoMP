"""A module to define base generators for particles and prior distributions.

This module includes abstract base classes for Particle generators (GeneratorBase),
parameter generators based on a set of prior distributions (ParamGeneratorBase),
and initial value generators based on a set of prior distributions (InitGeneratorBase).

HOW TO USE

When constructing a new particle filter, it is important to construct many independent
particles from the same set of prior distributions.  The classes in this sub-module
make this very simple; in general, unless you're defining your own model class, you
should only have to deal with the concrete implementations of these classes (e.g. 
the SEIHR generator in particle/SEIHR/SEIHR.py).
"""


from abc import ABC, abstractmethod
import numpy as np
import copy

class GeneratorBase(ABC):
    """Base class for generator objects.
    
    This class specifies the necessary functions for a concrete implementation
    (all of which are, in fact, attributes; please specify in a concrete
    implementation with the @property decorator).
    """
    
    def __init__(self, N, param_prior_dict, init_prior_dict):
        """Initialize a generator.
        
        We strongly suggest specifying custom defaults for concrete implementations
        of this class.
        
        Args:
            N (int): population size
            param_prior_dict (dict): dict of prior distributions (see examples in SEIHR sub-module)
            init_prior_dict (dict): dict of prior distributions (see examples in SEIHR sub-module)
        """
        self.N = N
        self.param_prior_dict = param_prior_dict
        self.init_prior_dict = init_prior_dict
    
    def generate(self, particle_id):
        """Generate a new particle with a set of randomly-generated values."""
        init_vals = self.init_generator(copy.deepcopy(self.init_prior_dict)).generate(self.N)
        
        return self.particle_class(
            particle_id = particle_id,
            init_vals = init_vals,
            param_generator = self.param_generator(copy.deepcopy(self.param_prior_dict))
        )
    
    @property
    @abstractmethod
    def param_generator(self):
        """Must be a concrete implementation sub-classed from ParamGeneratorBase"""
        pass
    
    @property
    @abstractmethod
    def init_generator(self):
        """Must be a concrete implementation sub-classed from InitGeneratorBase"""
        pass
    
    @property
    @abstractmethod
    def particle_class(self):
        """Must be a concrete implementation sub-classed from ParticleBase"""
        pass
    
class ParamGeneratorBase(ABC):
    """A class to control parameter generation.
    
    The purpose of this class is to establish a consistent API
    for specifying priors, some of which may be used to generate data
    partway through a simulation.  After a value is generated using
    a prior distribution, that distribution is replaced with
    the generated value so it can be reused in the future.
    
    To access the values of your parameters, see the .params attribute.
    
    For an example, please refer to the SEIHR_param_generator in the
    particle/SEIHR/SEIHR.py file for an example.
    """
    def __init__(self, params):
        """Initialize a ParamGenerator object (must be a concrete implementation.
        
        Args:
            params (dict): a dictionary of parameter priors (see SEIHR sub-module for examples)
        """
        self.params = dict(zip(
            self.param_ls,
            [
                {
                    'prior_ls' : params[x],
                    'cur' : None,
                    'end' : -1
                }
                for x in self.param_ls
            ]
        ))
        
    def generate(self, t_ls):
        """Generate a list of parameter values to use in forward simulation.
        
        This function is used to get/generate parameter values at a set of timestamps.
        
        This function returns a list of tuples indicating the parameters active during a period,
        along with the length of that period (the difference between two adjacent times in the
        t_ls).  Note that parameters are active according to the start time of a period, not the
        end time.
        
        Args:
            t_ls (list-like of numerics): list of times
            
        Returns:
            A list of tuples with values like
                ([<list of parameters>], dt)
        """
        return [
            (self.gen(t0), t1 - t0)
            for t0, t1 in zip(t_ls[:-1], t_ls[1:])
        ]
    
    @property
    @abstractmethod
    def param_ls(self):
        """The list of parameter names.
        
        This is the only abstract portion of this class.  It establishes
        a list of keys among the parameter dict you input and raises warnings if any of them
        are missing.  It also establishes a canonical ordering to your parameter values, which
        you should keep consistent in your particle class (in the `_update()` method).
        
        Note that you should use the @property decorator when specifying this in a concrete implementation.
        """
        pass
    
    
    def gen(self, t):
        """Get the parameters active at the given timestamp t.
        
        Args:
            t (numeric): the timestamp to use
            
        Returns:
            A list of parameter values at the given time.
        """
        p_out = []
        
        for param in self.param_ls:
            p_dict = self.params[param]
            if self.params[param]['end'] is None or t < self.params[param]['end']:
                p_out.append(self.params[param]['cur'])
            else:
                param_prior_ls = self.params[param]['prior_ls']
                ndx = [
                    t_ is None or t_ > t for (val,t_) in param_prior_ls
                ]
                
                ndx = min(np.where(ndx)[0])
                
                if callable(param_prior_ls[ndx][0]):
                    param_prior_ls[ndx] = (param_prior_ls[ndx][0](), param_prior_ls[ndx][1])
                p_out.append(param_prior_ls[ndx][0])
                
                self.params[param]['cur'] = param_prior_ls[ndx][0]
                self.params[param]['end'] = param_prior_ls[ndx][1]
                
        return p_out

class InitGeneratorBase(ABC):
    """A base class to generate initialization values for particles/simulations.
    
    This class operates very similarly to the ParameterGeneratorBase class, but is
    somewhat simpler because initialization values need only be generated once.
    """
    def __init__(self, compartment_priors):
        """Create an InitGenerator object (must be a concrete implementation, not Base class).
        
        Note that priors should be specified as percentages of the total, since the population
        size is given in the generate() method.
        
        Args:
            compartment_priors (dict): a dictionary of priors (see SEIHR sub-module for example)
        """
        self.compartment_priors = compartment_priors
        
    def generate(self, N):
        """Generate values from the priors with a total population size of N.
        
        Args:
            N (int): total population size
            
        Returns:
            A list of initial values for your compartments.s
        """
        p_out = []
        
        for compartment in self.compartment_ls:
            prior = self.compartment_priors[compartment]
            if callable(prior):
                p_out.append(prior())
            else:
                p_out.append(prior)
                
        tot = np.sum(p_out)
        
        tmp = np.array(p_out)/tot * N
        tmp = [int(x) for x in tmp]
        res = np.sum(tmp) - N
        tmp[0] -= res
        
        p_out = list(tmp)
                
        return p_out
    
    @property
    @abstractmethod
    def compartment_ls(self):
        """The list of compartment names.
        
        This is the only abstract portion of this class.  It establishes
        a list of keys among the initial value dict you input and raises warnings if any of them
        are missing.  It also establishes a canonical ordering to your compartment values, which
        you should keep consistent in your particle class (in the `_update()` method).
        
        Note that you should use the @property decorator when specifying this in a concrete implementation.
        """
        pass
    
    
   