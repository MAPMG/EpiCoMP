"""
The objects in this sub-module allow the user to combine the SEIQHR particle
object with the HCompartment object for the purpose of forecasting.

When fitting the particles, we do not fit the hospital length of stay
because this information is directly observable.  Instead, we employ
a Kaplan-Meier fit through the use of the distinct HCompartment object.
This makes for a superior fit, but also requires the user to employ
two completely separate objects and processes for forecasting.

The SEIQHR_combined_particle object in this sub-module allows the user
to re-combine these objects into a single logical unit when constructing
forecasts, and it abstracts the combination of this data into a single array.

The SEIQHR_forecast_manager is a meta-object that manages multiple
SEIQHR_combined_particle objects, allowing the user to create many parallel
forecasts and combine them into a 3-dimensional array.  This is useful when
constructing summary statistics about the possible paths for a forecast
(e.g. mean, median, confidence interval).
"""



import numpy as np
import pandas as pd
import copy

class SEIQHR_combined_particle():
    """An object to combine the SEIQHR particle object with the HCompartment."""
    
    def __init__(
        self,
        particle,
        h_compartment,
        admits,
        discharges,
        open_admit_dts
    ):
        """Initialize an SEIQHR_combined_particle.
        
        This method stores the (already fit) particle and h_compartment.
        In addition, it adjusts the current state of the particle in order
        to account for the most current data.  It uses the total admissions
        and discharges to date to align the current hospital census with
        the forecast; furthermore, it uses the ECDF of hospital length-of-stay
        to construct simulated discharge dates for all current hospitalizations,
        including generating these observations from the correct conditional
        ECDF based on the current LoS.
        
        This means that object initialization can sometimes take some time
        as all of the conditional probability distributions are evaluated.
        
        To reset the discharges and current state on the forecasting particle,
        use the _set_current function.
        
        Please note that this object assumes the current time to be the latest time
        from the particle object passed in (the .t attribute), and all times are
        assumed to be input as ints/floats relative to the start time of that particle.
        
        Args:
            particle (SEIQHR_Particle): A particle fit using the particle filter in this package
            h_compartment (HCompartment): An HCompartment object fit using this package
            admits (int): The total number of admissions to-date
            discharges (int): The total number of discharges to-date
            open_admit_dts (list): A list of the dates of admissions without corresponding discharges
        """
        
        self.hist_particle = particle.copy()
        self.h_compartment = h_compartment.copy()
        self.fut_particle = None
        self.base_fut_particle = None
        
        self._set_current(admits, discharges, open_admit_dts)
        
    def _set_current(self, admits, discharges, open_admit_dts):
        """
        This method reseets the base_fut_particle with the latest total admissions.
        It also reesets the h_compartment.  If this compartment already has the correct
        number of historical discharges, it leaves these untouched; otherwise, it resets
        these to the correct number of discharges, but occurring on the day before the start
        of the forecasting.
        
        Args:
            admits (int): The total number of admissions to-date
            discharges (int): The total number of discharges to-date
            open_admit_dts (list): A list of the dates of admissions without corresponding discharges
        """
        
        cur_h = admits
        cur_disc = discharges
        
        cur_r = np.sum(np.array(self.hist_particle.cur_vals)[[4,5]]) - cur_h
        
        if cur_r < 0:
            raise ValueError("Current simulation predicts too few R+H to account for current admissions.")
        
        cur_s = self.hist_particle.cur_vals[0]
        cur_e = self.hist_particle.cur_vals[1]
        cur_i = self.hist_particle.cur_vals[2]
        cur_q = self.hist_particle.cur_vals[3]
        
        t = self.hist_particle.t
        
        if self.h_compartment.get_history(t) != discharges:
            self.h_compartment.clear_history()
            self.h_compartment.add_history([t-1]*discharges)
            
        self.h_compartment.update(
            [1]*len(open_admit_dts), 
            open_admit_dts, 
            t - np.array(open_admit_dts)
        )
        
        self.base_fut_particle = self.hist_particle.__class__(
            particle_id = self.hist_particle.particle_id + '_fut',
            init_vals = np.array(
                [cur_s,cur_e,cur_i,cur_q,cur_h,cur_r]
            ),
            param_generator = copy.deepcopy(self.hist_particle.param_generator),
            t_init = t,
            w_t = self.hist_particle.w_t
        )
        
    def run_forecast(self, target_t, dt = 0.1):
        """Run a forecast until a target time (target_t) using a given timestap (dt).
        
        This function overwrites any previous forecasts made using this object.
        
        Args:
            target_t (numeric): the end-time for the simulation
            dt (numeric): the time-step of the simulation (default 0.1)
        """
        h_cur = self.base_fut_particle.cur_vals[4]
        self.h_compartment._discharges = self.h_compartment._discharges[:h_cur]
        
        self.fut_particle = self.base_fut_particle.step(target_t, 0, dt = dt)
        
        hist = self.fut_particle.history
        admits = hist[1:,4] - hist[:-1, 4]
        
        
        self.h_compartment.update(
            admits,
            self.fut_particle._t_history[:-1]
        )
        
    def replace_prior(self, new_prior_dict):
        
        old_prior_dict = copy.deepcopy(self.base_fut_particle.param_generator.params)
        
        for key in new_prior_dict:
            old_prior_dict[key]['end'] = -1
            old_prior_dict[key]['prior_ls'] = copy.deepcopy(new_prior_dict[key])

        self.base_fut_particle.param_generator.params = old_prior_dict
        
    @property
    def seiqhr_forecast(self):
        """A mask to call the underlying 'history' attribute of the forecasting particle"""
        if self.fut_particle is None:
            raise ValueError('Please run run_forecast() with a target time before attempting to access a forecast')
        return self.fut_particle.history
        
    @property
    def full_forecast(self):
        """
        A call to create and return a full forecast history including the H compartment.
        
        This forecast call alters the seiqhr_forecast call.  It adjusts the H compartment to be instantaneous
        instead of cumulative, and it adds in a new compartment (RH) as the last compartment value.  This new
        compartment collects patients that have been discharged from the hospital using the HCompartment
        simulation.
        """
        if self.fut_particle is None:
            raise ValueError('Please run run_forecast() with a target time before attempting to access a forecast')
            
        seiqhr_ = self.seiqhr_forecast
        disc = self.h_compartment.get_history(self.fut_particle._t_history)
        
        h = seiqhr_[:, 4] - disc
        
        return np.hstack([seiqhr_[:, [0,1,2,3]], h.reshape(-1,1), seiqhr_[:,4].reshape(-1,1), disc.reshape(-1,1), seiqhr_[:,5].reshape(-1,1)])
    
    
class SEIQHR_forecast_manager():
    """An object to manage multiple particle forecasts."""
    def __init__(
        self,
        particle_ls,
        h_compartment,
        admits, 
        discharges, 
        open_admit_dts
    ):
        
        """Initialize an SEIQHR_forecast_manager object.
        
        This method initializes a forecast manager; in addition, it also creates a collection of 
        independent SEIQHR_combined_particle objects, one for each particle in the given particle_ls
        by combining it with a copy of the input h_compartment object.
        
        Because this implicitly resets state on all of these combined particles to the currently-observed
        state, this operation can take some time to complete.
        
        Args:
            particle_ls (list of SEIQHR_Particle objects): the list of particles to use in forecasting
            h_compartment (HCompartment): An HCompartment object fit using this package
            admits (int): The total number of admissions to-date
            discharges (int): The total number of discharges to-date
            open_admit_dts (list): A list of the dates of admissions without corresponding discharges
        """
        
        self.comb_particles = [
            SEIQHR_combined_particle(
                particle,
                h_compartment,
                admits,
                discharges,
                open_admit_dts
            )
            for particle in particle_ls
        ]
    
    def run_forecast(self, target_t, dt = 0.1):
        """Run the .run_forecast() method on each of the particles controlled by the manager."""
        for x in self.comb_particles:
            x.run_forecast(target_t, dt = dt)
            
    def replace_priors(self, new_prior_dict):
        for part in self.comb_particles:
            part.replace_prior(new_prior_dict)

            
    @property
    def forecast(self):
        """Obtain the full_forecast attribute from each of the particles and concatenate into a 3-dimensional array."""
        return np.array([x.full_forecast for x in self.comb_particles])
