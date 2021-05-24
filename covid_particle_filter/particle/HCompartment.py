"""
The HCompartment sub-module contains classes to allow users to fit and simulate
a hospital Length-of-Stay (LoS) distribution.  This is performed by leveraging
the lifelines package's implementation of the Kaplan-Meier survival regression,
then using linear interpolation between the points to create a smoother
function.  Finally, we re-express the entire problem into a more convenient
domain for generating random variates (we express the interpolated ECDF as
a function of the probability domain from 0 to 1) and create a simple method
for forecasting.

This is a very simple and efficient method for simulating LoS for a large 
number of hospitalizations if we have a sufficient population to construct
a representative ECDF.  In practice, we suggest a minimum of several hundred
observations for this to be effective.  Otherwise, a parameterized distribution
may be superior.

Given the size of the data available to us, we felt that an empirical distribution
would offer us a better opportunity to effectively capture the long tail of our
Length-of-Stay distribution.

CONSTRUCTING PARAMETERIZED HCompartment OBJECTS

To construct a parameterized version of the HCompartment, one need only
substitute a new hosp_generator object in the HCompartment.  This can
be fit at run-time using the los array and the associated cens (censored)
array, or the fit() method can be passed.

What follows is an example of a parameterized HCompartment using an
exponential distribution.  The advantage here is that the memoryless
property would allow us to ignore the conditional input when
generating new observations.  For simplicity, we ignore
censored observations when fitting the exponential lambda.

Example parameterized HCompartment:

    import numpy as np
    
    class exp_hosp_generator(hosp_generator):
        def fit(self, los, cens):
            self.lambda = np.mean(los[~cens])
        def generate(self, N, conditional = 0):
            return np.random.exponential(self.lambda, N)
    
    class ExpHCompartment(HCompartment):
        def __init__(self):
            self.hosp_gen = exp_hosp_generator()
            self._discharges = []
"""


from lifelines import KaplanMeierFitter
from scipy import interpolate
import numpy as np
import pandas as pd
import copy

from collections.abc import Iterable

def is_iter(obj):
    return isinstance(obj, Iterable)

class hosp_generator():
    """A random variate generator following an empirical distribution fit using a linearized K-M curve."""
    def __init__(self):
        pass
    def fit(self, los, cens):
        """
        Fit a K-M curve, then use linear interpolation to smooth it.  Finally, recast it into 
        percentiles (by 1/1000) to allow for simple and efficient variate generation.
        
        Args:
            los (array): an array of length-of-stay observations
            cens (array): an array of booleans indicating whether each observation was right-censored
        """
        km = KaplanMeierFitter()
        
        if max(los) == max(los[cens]):
            los = np.array(list(los) + [max(los)*2 - max(los[~cens])])
            cens = np.array(list(cens) + [False])
        
        km.fit(np.array(los[~np.isnan(los)]), ~np.array(cens[~np.isnan(los)]))
        
        ecdf = km.cumulative_density_
        
        smooth_ecdf = interpolate.interp1d([0] + list(ecdf.KM_estimate), [0] + list(ecdf.index + 1))
        
        self._ecdf = smooth_ecdf(np.linspace(0.0,1,1000))
        
    def generate(self, N, conditional = 0):
        """Generate N random variates.
        
        Variates can be generateed using a conditional; in this case, the conditional should indicate the 
        amount of time already elapsed, and would therefore be used as a minimum to ensure that the remaining
        time is conditioned upon the elapsed time.
        
        If your conditional is larger than the max in the ECDF, we assume an exponential decline in the final 5%
        of the data.  Then, we employ the memoryless property to generate a discharge time using the conditional.
        
        Note: if conditional < 0, conditional is reset to 0.
        
        Args:
            N (int): the number of variates to generate
            conditional (numeric): the currently elapsed time (default 0)
            
        Returns:
            The total times to discharge (the variates generated) as an array.
            NOTE: this is not the *remaining* time to discharge; you must subtract conditional from this number
            to obtain the remaining time to discharge.
        """
        
        if conditional > 0:
            if conditional < max(self._ecdf):
                min_ = np.where(self._ecdf > conditional)[0][0]
            else:
                dists = self._ecdf[951:] - self._ecdf[950]
                m = dists.mean()
                return np.random.exponential(m, N) + conditional
        else:
            min_ = 0
        
        return self._ecdf[np.random.randint(min_,1000,N)]


class HCompartment():
    """An object to model and simulate hospital discharges."""
    def __init__(self):
        """Initialize an HCompartment object."""
        self.hosp_gen = hosp_generator()
        self._discharges = []
        
    def fit(self, los, cens):
        """Fit the underlying hosp_gen object on Length-of-Stay and censoring data.
        
        Args:
            los (array): an array of length-of-stay observations
            cens (array): an array of boolean indicators of whether each observation was right-censored.
        """
        self.hosp_gen.fit(los, cens)
        
    def generate(self, in_, conditional):
        """Generate a series of in_ variates using the corresponding conditional.
        
        Args:
            in_ (int): the number of admissions/variates to generate
            conditional (numeric/None): the conditional input to the hosp_generator object
            
        Returns:
            The generated LoS variates.
        """
        if conditional is None:
            return self.hosp_gen.generate(in_)
        else:
            return self.hosp_gen.generate(in_, conditional)
    
    def add_history(self, history_ls):
        """Add a history of discharges to this object.
        
        Args:
            history_ls (list-like): a list of discharge dates (in numeric terms, indexed to the simulation's zero-date)
        """
        self._discharges += list(history_ls)
        
    def clear_history(self):
        """Clear all discharge history."""
        self._discharges = []
        
    def update(self, in_, t, conditional = None):
        """Simulate a series of discharges and add them to the stored discharge history.
        
        Please note that, if in_ is an iterable, then the dimensions of t and conditional must match it (unless
        conditional is None).
        
        Args:
            in_ (int/iterable of ints): the number of discharges to simulate
            t (numeric/iterable of numerics): the admission times of the discharges to simulate
            conditional (numeric/iterable of numerics/None): the conditionals of the discharges to simulate
        """
        if is_iter(in_) and is_iter(t):
            if len(in_) != len(t):
                raise ValueError("The in_ and t inputs must either both be atomic or have the same dimension")
                
            if conditional is not None:
                if len(in_) != len(conditional):
                    raise ValueError("The in_ and conditional inputs must either both be atomic or have the same dimension")
                else:
                    lists = [
                        list(self.generate(in_[x], conditional[x]) + t[x])
                        for x in range(len(in_))
                    ]
                    self._discharges += [x for subl in lists for x in subl]
                    
                    
            else:
                lists = [
                    list(self.generate(in_[x], None) + t[x])
                    for x in range(len(in_))
                ]
                self._discharges += [x for subl in lists for x in subl]
                    
        elif is_iter(in_) or is_iter(t):
            raise ValueError("The in_ and t inputs must either both be atomic or have the same dimension")
        else:
            los = self.generate(in_, conditional = conditional)
            self._discharges += list(los + t)
        
    def __get_hist(self, t):
        """Return the cumulative discharges by the given timestamp."""
        return (np.array(self._discharges)<t).sum()
    
    
    
    def get_history(self, t_ls):
        """Retuurn the cumulative discharges by the given timestamp or iterable of timestamps.
        
        Args:
            t_ls (numeric/list of numerics): the times to get the desired cumulative discharge data
        
        Returns:
            If an iterable is received, returns an array of the cumulative discharges for each timestamp
            in the input iterable.  If an atomic is received, returns an integer indicating the number
            of cumulative discharges at that timestamp.
        """
        if is_iter(t_ls):
            return np.array(
                pd.concat([
                    pd.DataFrame({'admit_dt' : self._discharges}).groupby('admit_dt').size().reset_index(),
                    pd.DataFrame({'admit_dt' : t_ls, 0 : 0})
                ]).groupby('admit_dt').sum()[0].cumsum()[t_ls]
            )
        else:
            return self.__get_hist(t_ls)
    
    def copy(self):
        """Return a deep copy of self."""
        return copy.deepcopy(self)