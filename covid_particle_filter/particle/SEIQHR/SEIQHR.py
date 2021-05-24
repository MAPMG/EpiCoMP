"""A module to contain a particle for simulating or fitting a model with the following compartments:

    S -- Susceptible
    E -- Exposed
    I -- Infectious
    Q -- Quarantined
    H -- Cumulative Hospitalizations
    R -- Cumulative Non-Hospital Removals/Recoveries
    
This sub-module contains the particle object, along with a number of associated other objects,
including sample priors, the simulation update function, and a generator object.

PRIORS:

We also include several sample prior distributions; these should be interpreted as demonstrations,
not as suggestions.  These are the default distributions, but can also be explicitly imported
from this module.

Example Prior Distributions:

    def gen_h_rate():
        num_r = np.random.gamma(90/0.1, 0.1)
        num_h = np.random.gamma(10/1, 1)

        tot = num_r + num_h

        return num_h/tot

    sample_SEIQHR_param_prior = {
        'beta'  : [(lambda : np.random.gamma(3/.525, .075), 7), 
                   (lambda : np.random.gamma(3/.525, .075), 14), 
                   (lambda : np.random.gamma(2/.525, .075), 21),
                   (lambda : np.random.gamma(2/.525, .075), 28),
                   (lambda : np.random.gamma(2/.525, .075), 35),
                   (lambda : np.random.gamma(2/.525, .075), 42),
                   (lambda : np.random.gamma(2/.525, .075), 49),
                   (lambda : np.random.gamma(2/.525, .075), 56),
                   (lambda : np.random.gamma(2/.525, .075), 63),
                   (lambda : np.random.gamma(2/.525, .075), None)],
        'alpha' : [(lambda : np.random.gamma(0.3/0.02, 0.02), None)],
        'gamma' : [(lambda : np.random.gamma(0.15/0.01, 0.01), None)],
        'sigma' : [(lambda : np.random.gamma(0.15/0.01, 0.01), None)],
        'h_rate' : [(lambda : gen_h_rate(), None)]
    }

    sample_SEIQHR_init_prior = {
        'S'   : 99.9,
        'E'   : lambda : (np.random.poisson(9)+1)/1000,
        'I'   : 0,
        'Q'   : 0,
        'H'   : 0,
        'R'   : 0,
    }
    
These can be input to the generator as:

    #Note that N = 100000 would indicate a total
    #population size of 100k

    gen = SEIQHR_generator(
        N = 100000,
        param_prior_dict = sample_SEIQHR_param_prior, 
        init_prior_dict = sample_SEIQHR_init_prior
    )

Note that there is a defined API for defining how prior distributions interact.

Priors for parameters should be defined in a dictionary, with each key named after the associated
parameter and each value consisting of a list of tuples.  Each tuple should include a function which
returns a value for the parameter when called, along with a numeric value indicating when the 
algorithm should retire this value.  An entry of None indicates that the value should never
be retired.

For example, the prior:

    'beta'  : [(lambda : np.random.gamma(3/.525, .075), 7),
               (lambda : np.random.gamma(3/.525, .075), 14)]

in a parameter dictionary would indicate that the beta parameter should be generated twice.  First,
it should be generated at time 0 with a value from the distribution np.random.gamma(3/.525, .075).  Then,
at time 7, it should be re-generated with a new value from the same distribution.  Please note that
values are only generated when they are needed, so re-sampled particles will have different values
of the same parameter if their histories split prior to the new generation.

Please note that, while not demonstrated here, it is possiblee to specify an atomic numeric value
for a prior, e.g.

    'beta'  : [(2, 7)]

Also note that the definition above is slightly incorrect; at time 14, since there is no new prior,
the retirement time will be ignored and replaced with None.  The final value in a series
must always be None so a forward simulation won't error out.

Priors for Initialization values need only be specified as atomic functions or numerics in the value, e.g.

    'E'   : lambda : (np.random.poisson(9)+1)/1000
"""

import numpy as np

from covid_particle_filter.particle.particle_base import ParticleBase
from covid_particle_filter.particle.generator_base import GeneratorBase, ParamGeneratorBase, InitGeneratorBase

from scipy.stats import poisson, binom

def SEIQHR_update(vals, params, dt):
    """A function to perform the actual computation of forward-simulating an SEIQHR ODE.
    
    Note that there are several random steps to this simulation.
    
    S -> E
    
    The number of patients transitioning from S to E is drawn from a binomial distribution
    with probability:
        
        beta * I / N * dt
        
    (note that N denotes the total number of patients in the system)
        
    and with the population size:
        
        S
        
    Note that this has the same central value as a typical deterministic SEIR forward simulation,
    but allows for non-deterministic outcomes.
    
    E -> I
    
    The number of patients transitioning from E to I is also drawn from a binomial distribution,
    this one with probability:
        
        alpha * dt
        
    and with the population size:
        
        E
        
    Once again, this is a random generalization to the typical deterministic epidemiological model.
    
    I -> Q
    
    The number of patients transitioning from the actively infectious (I) compartment to the sick but
    quarantined (i.e. not infectious) Q compartment.  This transition follows a binomial distribution
    with probability:
        
        gamma*dt
    
    and with population size:
        
        I
    
    
    Q -> R+H
    
    The number of patients that are moving to *either* the hospital (H) or recovery (R) is also a
    binomial draw with probability
    
        sigma * dt
        
    drawn from a popuulation of size:
    
        Q
        
    We further sub-divide this population into patients going to H and patients going to R by drawing from
    a final binomial.  This final binomial has a probability
    
        h_rate (hospitalization rate)
        
    with population size:
    
        H+R
        
    This update schema allows for a stochastic generalization to the typical deterministic SEIR model (with R
    subdivided to allow us to measure some proportion of patients entering the hospital H).
    
    **Mathematical Note:**
    
    This parameterization is equivalent to an alternative parameterization which includes separate Poisson processes
    drawing patients from Q to H and Q to R.  This is explained by the following blog post (https://blog.jpolak.org/?p=1924).
        
    
    Args:
        vals (list): the current values of the compartments (SEIQHR)
        params (list): the current parameter values (beta, alpha, gamma, h_rate)
        dt (numeric): the size of the time-step to progress
        
    Returns:
        The new state of the system after forward-simulation as a list (SEIQHR)
    """
    S, E, I, Q, H, R = vals
    beta, alpha, gamma, sigma, h_rate = params
    
    N = np.sum(vals)
    
    p_ls = [
            beta*I/N*dt,
            alpha*dt,
            gamma*dt,
            sigma*dt
        ]
    
    n_ls = [
        S,
        E,
        I,
        Q
    ]
    
    new_e, new_i, new_q, q_out = np.random.binomial(n_ls, p_ls)
    new_h = np.random.binomial(q_out, h_rate)
    new_r = q_out - new_h
    
    S = S - new_e
    E = E + new_e - new_i
    I = I + new_i - new_q
    Q = Q + new_q - q_out
    H = H + new_h
    R = R + new_r
    
    return [S, E, I, Q, H, R]

class SEIQHR_param_generator(ParamGeneratorBase):
    """A parameter generator for the SEQIHR model.
    
    Note that this parameter generator requires the following parameters:
    
        beta -- the instantaneous infectiousness rate (must be > 0)
        alpha -- the inverse of the average incubation period (must be > 0)
        gamma -- the inverse of the average infectiousness period (must be > 0)
        sigma -- the inverse of the average quarantined period (must be > 0)
        h_rate -- the average hospitalization rate (must be between 0 and 1)
    """
    @property
    def param_ls(self):
        return ['beta', 'alpha', 'gamma', 'sigma', 'h_rate']
    
class SEIQHR_init_generator(InitGeneratorBase):
    """An initial value generator for the SEIQHR model.
    
    Note that this generator must construct the following compartments:
    
        S -- susceptible
        E -- exposed
        I -- infectious
        Q -- quarantined
        H -- hospitalized
        R -- removed/recovered
    """
    @property
    def compartment_ls(self):
        return ['S', 'E', 'I', 'Q', 'H', 'R']
    

def gen_h_rate():
    """A helper function to create an h_rate centered on 0.9."""
    num_r = np.random.gamma(90/0.1, 0.1)
    num_h = np.random.gamma(10/1, 1)
    
    tot = num_r + num_h
    
    return num_h/tot

sample_SEIQHR_param_prior = {
    'beta'  : [(lambda : np.random.gamma(3/.525, .075), 7), 
               (lambda : np.random.gamma(3/.525, .075), 14), 
               (lambda : np.random.gamma(2/.525, .075), 21),
               (lambda : np.random.gamma(2/.525, .075), 28),
               (lambda : np.random.gamma(2/.525, .075), 35),
               (lambda : np.random.gamma(2/.525, .075), 42),
               (lambda : np.random.gamma(2/.525, .075), 49),
               (lambda : np.random.gamma(2/.525, .075), 56),
               (lambda : np.random.gamma(2/.525, .075), 63),
               (lambda : np.random.gamma(2/.525, .075), None)],
    'alpha' : [(lambda : np.random.gamma(0.3/0.02, 0.02), None)],
    'gamma' : [(lambda : np.random.gamma(0.15/0.01, 0.01), None)],
    'sigma' : [(lambda : np.random.gamma(0.15/0.01, 0.01), None)],
    'h_rate' : [(lambda : gen_h_rate(), None)]
}

sample_SEIQHR_init_prior = {
    'S'   : 99.9,
    'E'   : lambda : (np.random.poisson(9)+1)/1000,
    'I'   : 0,
    'H'   : 0,
    'Q'   : 0,
    'R'   : 0,
}

class SEIQHR_Particle(ParticleBase):
    """Particle class for SEIQHR Particle.
    
    This particle function assumes that we fit on hospitalizations and use the SEIQHR_update
    function for forward simulation.
    """
    def _update(self, vals, params, dt):
        """Mask for SEIQHR_update."""
        return SEIQHR_update(vals, params, dt)
    def _evaluate(self, eval_value, cur_vals, prev_vals):
        """Evaluate a sample to set the sampling weight.
        
        This function returns the particle's sampling weight by evaluating the probability
        of the deltas in the current simulation (cur_vals - prev_vals) obtaining the hospitalizations
        observed over that period (eval_value).
        
        In this case, that probability is equal to the binomial probability of obtaining eval_value
        from a population equal to the number of patients transitioning out of the I compartment (we
        ignore the split here between the H and R compartments).  We assume a hospitalization
        probability of h_rate.
        
        This means that the observed input to the H compartment may be far from the simulated input to the H 
        compartment, even if the probability of obtaining the data was high.  Over many particles, this should
        create an appropriate confidence interval around the observed hospitalizations; furthermore, we can easily
        reset these values at a later point (which we do using the CombinedSEIQHR classes).
        
        Args:
            eval_value (int): the number of hospitalizations observed over the period
            cur_vals (list of ints): the values of the compartments at the end of the evaluation period
            prev_vals (list of ints): the values of the compartments at the start of the evaluation period
            
        Returns:
            The new sampling weight to be evaluated (the current self.w_t * the probability of obtaining the new data).
        """
        delta_H_plus = eval_value
        
        delta_plus_sim = np.sum(np.array(cur_vals)[[4,5]]) - np.sum(np.array(prev_vals)[[4,5]])
        
        
        prob_plus = binom.pmf(delta_H_plus, delta_plus_sim, self.param_generator.params['h_rate']['cur'])
        
        w_t = self.w_t * (prob_plus+10**(-4))
        return w_t

class SEIQHR_generator(GeneratorBase):
    """A generator for SEIQHR_Particle objects."""
    def __init__(
        self, 
        N,
        param_prior_dict = sample_SEIQHR_param_prior, 
        init_prior_dict = sample_SEIQHR_init_prior,
        particle_class = SEIQHR_Particle
    ):
        
        super().__init__(N, param_prior_dict, init_prior_dict)
        self.part = particle_class
        
    @property
    def param_generator(self):
        return SEIQHR_param_generator
    
    @property
    def init_generator(self):
        return SEIQHR_init_generator
    
    @property
    def particle_class(self):
        return self.part
    