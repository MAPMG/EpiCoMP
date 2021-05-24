"""
A module to implement a genetic particle filter.

The core importable element of this module is the particle_filter function.
This function has an identical API to the traditional particle filter function,
but instead uses a Genetic Algorithm for resampling.  Please refer to:

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5421656/

Please note that the genetic recombination step assumes a very particular set of internal
APIs for your particles.  You must use the param generators as implemented in the `particle`
module, and your particle history must be a list of arrays (as it is by default).

Also, note that the genetic resampling step is back-applied to historical values.  This maintains
the core concept of the algorithm, but may lead to confusion when fitting.  We recommend snapshotting
intermediate fits if you employ parameter changepoints.
"""

from covid_particle_filter.particle.SEIHR import SEIHR_generator
import numpy as np
import warnings
import copy

#Helper functions for the genetic algorithm
def combine_element(elem_1, elem_2, b):
    if callable(elem_1[0]):
        return elem_1
    else:
        return (elem_1[0]*b + elem_2[0]*(1-b), elem_1[1])

def combine_params(param_gen_1, param_gen_2, b):
    
    param_dict = {}
    
    for key in param_gen_1.params.keys():
        param_dict[key] = {
            'prior_ls' : [combine_element(x, y, b) for x,y in 
                          zip(param_gen_1.params[key]['prior_ls'], param_gen_2.params[key]['prior_ls'])],
            'cur' : param_gen_1.params[key]['cur']*b + param_gen_2.params[key]['cur']*(1-b),
            'end' : param_gen_1.params[key]['end']
        }
        
    pgen = copy.deepcopy(param_gen_1)
    pgen.params = param_dict
    
    return pgen


def combine_particles(particle_1, particle_2):
    
    b = particle_1.w_t / (particle_1.w_t + particle_2.w_t)
    
    history = [(np.array(x)*b + np.array(y)*(1-b)).round(0) for x, y 
               in zip(particle_1._history, particle_2._history)]
    
    cur_vals = list(
        (np.array(particle_1.cur_vals)*b + np.array(particle_2.cur_vals)*(1-b)).round(0)
    )
    
    t_init = particle_1.t
    
    t_history = particle_1._t_history
    
    pgen = combine_params(particle_1.param_generator, particle_2.param_generator, b)
    
    return particle_1.__class__(
            particle_id = 'comb_',
            init_vals = cur_vals,
            param_generator = pgen,
            t_init = t_init,
            w_t = particle_1.w_t,
            _history = history,
            _t_history = t_history
        )
    



def particle_filter(
    h_obs,
    n_particles = 100000,
    dt = 0.1,
    gen = SEIHR_generator(100000),
    particles = None,
    s_min = None
):
    
    """Perform a particle filtration with a genetic resampler.
    
    Please refer to the documentation for the traditional particle filter for more detail on how this algorithm
    works at a high level.  We only discuss the core differences between the two algorithms in this documentation.
    
    In order to employ a genetic resampling, we have used several helper functions.  If necessary, these could be reimplemented;
    they can be individually examined by importing:
    
    * combine_particles
    * combine_params
    * combine_element
    
    from this module.  
    
    At a high level, the genetic resampling operates by identifying the top `s_eff` particles.  Then, if this number is below
    `s_min`, we separate out all other particles.  We retain the top `s_eff` particles as-is; in order to get back to `n_particles`
    particles, we create a set of genetically recombined particles.  Each recombined particle is a combination of one particle from the
    top set and one particle from the bottom set.  All values (parameters, current and historical values) are combined by taking the
    weighted average of that value from the two particles.  The weights for the weighted average are the re-normalized weights from the
    two particles.
    
    Args:
        h_obs (list): a list of tuples (t,y) that indicates the evaluation value (y) occurring
        between the previous pair's t value and the current t value (by default, hospital admissions over that period)
        n_particles (int): the number of particles to create (default 10000)
        dt (numeric): the step size for the simulation (NOT for the evaluation steps; default 0.1)
        gen (generator): a generator to construct a set of particles (default SEIHR_generator from the particle module)
        particles (list of instantiated Particle objects): a list of particles that have already been partially fitted
        (overrides the generator and continues fitting these particles)
        s_min (float): minimum effective sample size that triggers resampling (defaults to 0.9*n_particles if None/not specified)
    Returns:
        A list of n_particles particles which have survived the filtration process.
    """
    
    if particles is not None:
        n_particles = len(particles)
        t_prev = particles[0].t
        if any([t_ <= t_prev for t_, y in h_obs]):
            warnings.warn("Dropping observations with time below current simulation time")
            h_obs = [(t_, y) for t_, y in h_obs if t_ > t_prev]
            
        particles = [p.copy() for p in particles]
        
    else:
        particles = [
            gen.generate(i) for i in range(n_particles)
        ]
        t_prev = 0
    
    if s_min is None:
        s_min = n_particles*0.9
    
    for t, y in h_obs:
        tmp_particles = [
            x.step(t, i, dt = dt, eval_value = y) for x, i in zip(particles, range(len(particles)))
        ]
        
        pre_weights = [x.w_t for x in tmp_particles]
        weights = np.array(pre_weights)/np.sum(pre_weights)
        s_eff = 1/np.sum(weights**2)
        
        for x, w in zip(tmp_particles, weights):
            x.w_t = w
            
        print(s_eff)
        
        if s_eff < s_min:
            
            asis = np.array(weights).argsort()[-int(s_eff):]
            
            resamp = np.array(weights).argsort()[:-int(s_eff)]
            
            particles = [tmp_particles[x] for x in asis] + \
                [combine_particles(tmp_particles[x], tmp_particles[y])
                 for x,y in zip(
                     np.random.choice(asis, size = n_particles - int(s_eff), replace = 1),
                     np.random.choice(resamp, size = n_particles - int(s_eff), replace = 1)
                 )
                ]
            
            for x in particles:
                x.w_t = 1/len(particles)
                
        else:
            particles = tmp_particles
        
    return particles