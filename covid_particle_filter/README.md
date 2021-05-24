# Module Reference

This module includes two sub-modules.  The `filter` sub-module includes code necessary for implementing a particle filter.  It is completely particle-agnostic, but assumes certain APIs are available with any particle objects it tries to fit.  The `particle` sub-module is completely divorced from the filtering methodology, and can be used independently as a framework for constructing simulations based on systems of differential equations and arbitrary numerical priors.

Please note that, when using this package, you must explicitly import from the `filter` and `particle` sub-modules because of their relative independence.  The top-level `__init__.py` file does not include automatic imports of either sub-module.