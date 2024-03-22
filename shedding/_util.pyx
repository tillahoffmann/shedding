from libc cimport math
cimport cython
cimport scipy.special.cython_special as cspecial


cdef:
    double _LN_PDF_CONSTANT = math.log(2 * math.M_PI) / 2
    double _SQRT2 = math.sqrt(2)


@cython.ufunc
@cython.cdivision(True)
cdef double gengamma_lpdf(double q, double mu, double sigma, double logx):
    cdef:
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
    return value


@cython.ufunc
@cython.cdivision(True)
cdef double gengamma_lcdf(double q, double mu, double sigma, double logx):
    cdef:
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

    if value == 0:
        return -math.INFINITY
    else:
        return math.log(value)


@cython.ufunc
@cython.cdivision(True)
cdef double gengamma_loc(double q, double sigma, double mean):
    if q == 0:
        value = sigma * sigma / 2
    elif q < 1e-6:
        value = sigma * sigma / 2 - q * sigma / 6 * (sigma * sigma + 3)
    else:
        value = 1 / (q * q)
        value = math.lgamma(value + sigma / q) - math.lgamma(value) - math.log(value) * sigma / q
    return math.log(mean) - value
