cimport numpy as np
from cyfunc cimport get_value, set_value, create_signature, register_cyfunc
from libc cimport math
cimport cython
cimport scipy.special.cython_special as cspecial


cdef:
    double _LN_PDF_CONSTANT = math.log(2 * math.M_PI) / 2
    double _SQRT2 = math.sqrt(2)


@cython.cdivision(True)
cdef void gengamma_lpdf_d(char** args, void* data):
    cdef:
        double q = get_value[double](args, 0, 0)
        double mu = get_value[double](args, 1, 0)
        double sigma = get_value[double](args, 2, 0)
        double logx = get_value[double](args, 3, 0)
        double z = (logx - mu) / sigma
        double logsigma = math.log(sigma)
        double value, s, a, logq
    # We are exactly in the lognormal regime.
    if q == 0:
        value = - _LN_PDF_CONSTANT - logsigma - z * z / 2 - logx
    # We are in the dodgy regime where numerics are tricky.
    elif q < 1e-6:
        s = math.expm1(q * z) / q
        value = - _LN_PDF_CONSTANT - logsigma - s * s / 2 + q * z - logx
    # We can evaluate the log pdf normally.
    else:
        a = 1 / (q * q)
        logq = math.log(q)
        value = logq - logsigma - math.lgamma(a) - 2 * a * logq + (a * q / sigma - 1) * logx - \
            a * (mu * q / sigma + math.exp(q * z))

    set_value(args, 4, value)


@cython.cdivision(True)
cdef void gengamma_lcdf_d(char** args, void* data):
    cdef:
        double q = get_value[double](args, 0, 0)
        double mu = get_value[double](args, 1, 0)
        double sigma = get_value[double](args, 2, 0)
        double logx = get_value[double](args, 3, 0)
        double z = (logx - mu) / sigma
        double value

    if q == 0:
        value = (1 + math.erf(z / _SQRT2)) / 2
    elif q < 1e-6:
        s = math.expm1(q * z) / q
        value = (1 + math.erf(s / _SQRT2)) / 2
    else:
        value = 1 / (q * q)  # Using `value` as a substitute for a to save memory.
        value = cspecial.gammainc(value, value * math.exp(q * z))

    set_value(args, 4, math.log(value))


@cython.cdivision(True)
cdef void gengamma_loc_d(char** args, void* data):
    cdef:
        double q = get_value[double](args, 0, 0)
        double sigma = get_value[double](args, 1, 0)
        double mean = get_value[double](args, 2, 0)

    if q == 0:
        value = sigma * sigma / 2
    elif q < 1e-6:
        value = sigma * sigma / 2 - q * sigma / 6 * (sigma * sigma + 3)
    else:
        value = 1 / (q * q)
        value = math.lgamma(value + sigma / q) - math.lgamma(value) - math.log(value) * sigma / q
    set_value(args, 3, math.log(mean) - value)


signature = create_signature([float, float, float, float], [float], gengamma_lpdf_d, <void*>0)
gengamma_lpdf = register_cyfunc('gengamma_lpdf', '[gengamma_lpdf docstring]', [signature])

signature = create_signature([float, float, float, float], [float], gengamma_lcdf_d, <void*>0)
gengamma_lcdf = register_cyfunc('gengamma_lcdf', '[gengamma_lcdf docstring]', [signature])

signature = create_signature([float, float, float], [float], gengamma_loc_d, <void*>0)
gengamma_loc = register_cyfunc('gengamma_lloc', '[gengamma_loc docstring]', [signature])

