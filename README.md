# EpiCoMP: **Epi**demiological **Co**mpartment **M**odel **P**article Filters

[Anthony Finch](anthony.j.finch@kp.org)
[Alexander Crowell](alexander.m.crowell@kp.org)
[Michael Horberg](michael.a.horberg@kp.org)

Welcome to EpiCoMP.

This package has been designed to expose a simple, user-friendly interface to epidemiological particle filtering.  Please see <paper_link> for the accompanying publication.

## Epidemiological Modeling

This package is structured around epidemiological compartment models such as the SEIR (**S**usceptible, **E**xposed, **I**nfectious, **R**emoved) model; however, there is no reason that our particle filtering framework cannot be extended to incorporate a completely different system of differential equations.

For more information on basic compartment models, please see the following papers/textbooks [Compartmental Models in Epidemiology](https://link.springer.com/content/pdf/10.1007%2F978-3-540-78911-6_2.pdf).



The original motivation for building this package was to construct models of COVID-19 that could be updated continuously, with minimal training time.  For this reason, we use a *Particle Filtering* approach.

## Particle Filtering

### Background

Particle filtering is an online learning method similar to the Kalman filter.  We begin with a group of particles initialized from a set of rules (our prior distribution).  Then, at each timestep, we resample particles based on how likely they were to produce the measurements obtained.  For a more detailed treatment on basic particle filtering, please see [Machine Learning: A Probabilistic Perspective](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf).

There are several advantages to particle filtering over other fitting methods.  Particle filters are

1. computationally efficient to update because they employ online learning
2. an efficient Bayesian sampling algorithm 
3. highly effective at sampling hidden (unobserved) states in Hidden Markov Models (as in an SEIR model where we only observe a certain percentage of infectious, hospitalized, or mortal cases).

Alternative methods for model fitting of this type include:

* Markov Chain Monte Carlo (MCMC), Gibbs, Metropolis-Hastings, or other sampling techniques (Bayesian samplers)
* Nonlinear Least Squares, Stochastic Gradient Descent, or others (Maximum Likelihood estimators)
* Expert parameter estimation (essentially no fitting)

These are all effective techniques and have certain advantages or disadvantages in comparison to particle filters.  It is possible to extend particle filters using one or more of these methods to improve overall performance.  The R package [POMP](https://cran.r-project.org/web/packages/pomp/pomp.pdf) adds many such improvements, although there is no well-established Particle Filtering package in Python to the best of our knowledge.

Our goal with this package was to construct a simple way to employ Particle Filters to fit epidemiological compartment models in Python.  We did not include more advanced fitting modalities (e.g. MCMC resampling) and have emphasized the ability to fit and forecast these models in an efficient online fashion (i.e. without the need to reuse old data as new data surfaces to continue model fitting).

### Algorithm Overview

The general method of a particle filter is to:

1. Initialize N separate 'particles' (independent simulations)
2. Update each particle according to an evolutionary/simulation rule
3. Calculate the probability of obtaining a measurement (the true data) from each particle
4. Resample (with replacement) particles from the population based on their relative likelihoods

Note that this resampling step is frequently set so that it only engages after the diversity of particle probabilities reaches a particular threshold.  For instance, if all particles are equally likely, we would not want to resample our particles because we would be needlessly eliminating 'good' particles, which is computationally inefficient.  Over many iterations, this algorithm will typically converge to a high-fidelity estiamte of system state and parameters.

We have also included an alternative Genetic Particle Filter algorithm, as described by [Li, et al.](https://pubmed.ncbi.nlm.nih.gov/28350341/).  This alternative algorithm uses a Genetic Algorithm to adjust lower-quality particles instead of elimintating them.  For simplicity, we omit the mutation step of this algorithm, although it is simple to add it back into the algorithm (see /covid_particle_filter/filter/filter_ga.py).


## Package Overview

Our package splits this algorithm into two separate sub-modules, which are individually editable.

The `particle` submodule contains definitions for an abstract particle class, which can be sub-classed to construct arbitrary systems of differential equations for numerical simulation.  We include a concrete implementation of our own sub-class for SEIHR and SEIQHR models (which add a **H**ospitalization compartment to the traditional SEIR model and a **Q**uarantined compartment to that model, respectively); however, we also provide examples and documentation to allow users to define their own particles.

The `filter` submodule is system-agnostic; it calls a specific set of attributes and methods from objects with a particle super-class.  As long as a particle is correctly implemented, the algorithms implemented herein will fit it.  Please note that we include two filtering algorithms.  In addition to the traditional particle filter (as described above), we also build a *Genetic* Particle Filter class, which uses a genetic resampling step instead of a traditional resampling step.  This method is closely related to the genetic resampling proposed by [Li, et al.](https://pubmed.ncbi.nlm.nih.gov/28350341/), although we do not incorporate a mutation step.  Our algorithm is described in detail in [our paper](link).

### Design Considerations

We have designed this package to incorporate Object-Oriented Programming as much as practicable.  New particle implementations should be sub-classed from the ParticleBase class (available in <package_name>/particle/particle_base.py).  We incorporate the use of factory objects (called generators in our module) in order to ensure that particles are generated with different parameter values according to the same set of priors.  In order to create a full particle definition, you will also need to define a custom generator which you can hand to the particle filter.

### Additional Functionality

In addition to the functionality described above, we have also implemented an 'H Compartment' class and a 'Combined' particle.  This is a useful extension to the SEIHR model, which allows the user to empirically estimate the distribution for hospital length-of-stay and to explicitly model hospital census over time.  This acts as an extension to the basic model definition explored in our [paper](link).

### Tutorials

We include a series of simple tutorials (using Jupyter Notebooks) in the docs sub-directory.