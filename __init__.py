"""
For examples of dadi's usage, see the examples directory in the source
distribution.

Documentation of all methods can be found in doc/api/index.html of the source
distribution.
"""
import logging
logging.basicConfig()

import Demographics1D
import Demographics2D
import Godambe
import Inference
import Integration
import Misc
import Numerics
import PhiManip

# import the triallele modules - numerics, integration, demographics
try:
    import Triallele.numerics, Triallele.integration, Triallele.demographics
except ImportError:
    pass

# Protect import of Plotting in case matplotlib not installed.
# 
# Comment out these lines if using on the cluster. -Bernard
try:
    import Plotting
except ImportError:
    pass

# We do it this way so it's easier to reload.
import Spectrum_mod 
Spectrum = Spectrum_mod.Spectrum

# When doing arithmetic with Spectrum objects (which are masked arrays), we
# often have masked values which generate annoying arithmetic warnings. Here
# we tell numpy to ignore such warnings. This puts greater onus on the user to
# check results, but for our use case I think it's the better default.
import numpy
numpy.seterr(all='ignore')