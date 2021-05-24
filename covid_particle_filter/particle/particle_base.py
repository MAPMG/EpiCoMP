"""
A module with abstract base classes to enable users to construct their own particles.

NOTE: when implementing a new particle class, please ensure that parameters and compartment
variables are always listed in the same order across both this class and the generators you create.
"""

from abc import ABC, abstractmethod
import numpy as np
import copy



class ParticleBase(ABC):
    """An object to simulate an extended SEIR model.  Can be used for Particle Filtering.
    
    At a high-level, this is a wrapper to manage stochastic ODE simulations.  It includes basic utilities to
    step a simulation forward.  In general, it should not be necessary to edit these methods.  To create a custom
    simulation, you need only implement `_update` and `_evaluate` methods.  These methods define, respectively, how
    to update the state of the system (which is maintained as an arbitrary vector in the `_history` and `cur_vals` attributes).
    """
    def __init__(
        self, 
        particle_id,
        init_vals,
        param_generator,
        t_init = 0,
        w_t = 1,
        _history = None,
        _t_history = None
    ):
        """Create a particle object.
        
        Args:
            particle_id (str): an id - useful for identifying ancestor particles
            cur_vals (list): a list of numeric starting values for each compartment
            param_generator (func): a function that takes a list of time values and returns a list of tuples
            the first element of which is a list of params and the second element of which is a time delta (dt)
            w_t (numeric): the current weight of the particle (as determined by condensation algorithm; default 1)
            _history (list of lists): a list-of-lists history (default None)
            _t_history (list): a list of previous values of t (default None)
        """
        
        self.particle_id = particle_id
        self.cur_vals = init_vals
        self.param_generator = param_generator
        self.t = t_init
        self.w_t = w_t
        self._history = _history
        if _t_history is not None:
            self._t_history = _t_history
        else:
            self._t_history = [t_init]
        self.prev_vals = None
        
    @abstractmethod
    def _update(self, vals, params, dt):
        """An abstract method to update the current state of the simulation.
        
        This method takes 3 arguments (additional arguments are not supported without editing
        the class's base `_step_forward` and `step` methods).
        
        This method should include only a single incremental step forward; looping over these steps is abstracted
        to other methods.
        
        As an example of this kind of function, please refer to the `SEIHR_update` function in the particle.SEIHR.SEIHR.py
        folder.
        
        Args:
            vals (list): a list of the current values of the system
            params (list): a list of the parameters used to update the system
            dt (numeric): the step-size to use in the simulation
            
        Returns:
            A list of updated variable values; should be in the same order as they are in `vals`.
        """
        pass
    
    @abstractmethod
    def _evaluate(self, eval_value, cur_vals, prev_vals):
        """An abstract method to update the 'weight' of the particle (for particle filtering).
        
        This function **should not** update the current value of its own weight (stored in self.w_t).  Instead, it
        should return the updated weight value.  We strongly recommend that you do not alter any attributes of your object
        inside this method.
        
        This function is typically used in particle filtering to generate the relative weights of particles to decide which
        particles should be filtered or mutated.  The algorithm should typically require the evaluation value (the data
        you're comparing against), the current value, and the previous value.  In our SEIHR models, we typically compare
        hospital admissions over the given time period (the difference between current H and previous H) with the observed
        admissions over that same period (given by `eval_value`).  For an example of this kind of implementation,
        please refer to the `_evaluate()` method in our SEIHR particle in particle/SEIHR/SEIHR.py.
        
        **PLEASE NOTE** we strongly recommend that you compute the new weight by multiplying the old weight by the
        probability of obtaining the current observations; however, we have not implemented this in any of the 
        wrapper functions in order to leave this open to advanced users who may choose not to do this.  By default,
        particle filtering algorithms should use this.
        
        Please also note that we strongly recommend that you add a small positive value to your multiplicative weight
        to prevent overflow errors that could result in negative probabilities.
        
        Ex:
        
            w_t = self.w_t * (prob_plus+10**(-4))
            return w_t
        
        Args:
            eval_value (int): the observed data to compare against (in our case, hospitalizations; need not be atomic)
            cur_vals (list of numerics): the values of the compartments at the end of the evaluation period
            prev_vals (list of numerics): the values of the compartments at the start of the evaluation period
            
        Returns:
            The new sampling weight to be evaluated (the current self.w_t * the probability of obtaining the new data).
        """
        pass
        
    @property
    def history(self):
        """Return the history as an array; history is stored as a nested list of lists because computing
        the array incrementally is extremely inefficient when simulating."""
        if self._history is None:
            return None
        history = np.vstack([np.vstack(x) for x in self._history])
        
        return history
        
    def copy(self):
        """Return a deep copy of self."""
        return copy.deepcopy(self)
        
    def _step_forward(self, t_ls):
        """Step forward to each value t in t_ls.
        
        This method abstracts the iteration of a series of updates to the system state using dynamics governed
        by self.update().
        
        Note that updates performed using this function *do not* perform evaluation steps/resampling.
        
        Args:
            t_ls (list-like): a list of values of time t; should start with the current time
        Returns:
            A list of lists and a flat list.  Each sub-list in the list-of-lists contains the value of the system
            at that time-step.  Each element in the flat list contains the associated timestamp of the system.
            
            Note that both lists will be shorter than the input t_ls by 1, since the first element in t_ls
            is assumed to be the current time.
        """
        vals = [int(x) for x in self.cur_vals]
        
        steps = []
        
        for params, dt in self.param_generator.generate(t_ls):
            vals = self._update(vals, params, dt)
            steps.append(vals)
            
        return steps, t_ls[1:]
        
    def step(self, t, m, dt, eval_value = None):
        """Step a simulation forward to a specified timestamp.
        
        This method returns a new particle.  The only possible alteration to self
        is if the step to t crosses to a new beta (in which case, that beta is recorded
        permanently in self.breakpoints).
        
        The new particle has all the same parameters as self, but includes the new history from
        stepping forward.  The value of cur_vals is deprecated to the new particle's prev_vals
        and the latest value from the new history parameter is set as cur_vals.  The new
        particle's particle_id is set to the current particle_id + the new digit m.
        
        Finally, the candidate step is compared to the eval_value.  If eval_value is None, we always
        accept the proposal.  Otherwise, we call the evaluate() function and only return if the result
        is True.  Otherwise, we discard the proposal and return None.
        
        Args:
            t (numeric): end time for step-forward
            m (int): the new digit to add to the particle_id
            dt (numeric): the size of the incremental sub-steps
            eval_value (numeric/None): the observed data against which to compare
        """
        t_diff = t - self.t
        init_vals = self.cur_vals
        t_ls = np.linspace(self.t, t, int(t_diff/dt) + 1)
        res, new_t_ls = self._step_forward(list(t_ls))
        
        cur_vals = res[-1]
        prev_vals = init_vals
        
        if eval_value is None:
            new_w_t = self.w_t
        else:
            new_w_t = self._evaluate(
                eval_value = eval_value, 
                cur_vals = cur_vals, 
                prev_vals = prev_vals
            )
            
        ints = np.unique([int(x) for x in new_t_ls])
        if ints[0] == int(self.t):
            ints = ints[1:]
        
        keep = [np.where(np.array(new_t_ls).astype(int) == x)[0][0] for x in ints]
        
        if not self.history is None:
            history = self._history + [list(np.array(res)[keep])]
        else:
            history = [[self.cur_vals]] + [list(np.array(res)[keep])]
            
        new_particle = self.__class__(
            particle_id = str(self.particle_id) + str(m),
            init_vals = cur_vals,
            param_generator = copy.deepcopy(self.param_generator),
            t_init = new_t_ls[-1],
            w_t = new_w_t,
            _history = history,
            _t_history = self._t_history + list(np.array(new_t_ls)[keep])
        )
        
        return new_particle