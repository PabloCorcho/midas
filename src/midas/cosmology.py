from astropy.cosmology import FlatLambdaCDM
from numpy import array, logspace

# Cosmological model used for computing angular diameter distance...
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Mapping between scale factor H(t)/H0 to look-back times (stellar ages)
scale_f = array(list(map(cosmo.scale_factor, logspace(-5, 2, 100))))
age_f = array(
    [cosmo.lookback_time(z_i).value for z_i in logspace(-5, 2, 100)])