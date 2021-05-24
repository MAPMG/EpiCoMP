# Filters

This sub-module includes two particle filters.  `filter.py` includes a traditional particle filter, which employs resampling whenever a system's effective sample size after an iteration is below a specified threshold.  On the other hand, `filter_ga.py` employs a genetic algorithm for resampling, instead of the traditional approach.  We employ the same API for each of these filters, so it should be trivial to switch between the two methods, e.g.

    from covid_particle_filter.filter.filter import particle filter as pf
    
    ...
    
    fit_particles = pf(h_obs = data, gen = my_generator)
    
Conversely, we can use the genetic particle filter as:

    from covid_particle_filter.filter.filter_ga import particle filter as pf
    
    ...
    
    fit_particles = pf(h_obs = data, gen = my_generator)

Aside from the mildly altered import statement, everything else will be the same.

# Importing

To avoid name collisions between different types of filters, we require that you specify the full path to the filter.  There are two functions users are likely to employ from this sub-module, which can be imported as:

    # Imports the traditional particle filter
    from covid_particle_filter.filter.filter import particle filter as pf
    
    # Imports the genetic particle filter
    from covid_particle_filter.filter.filter_ga import particle filter as pf
    
For more detail on specific APIs, please refer to the documentation in the code itself.