from numba import njit, float64, float32


@njit(fastmath=True, cache=True)  # ([float64(float64,float64),float32(float32,float32)])
def ForceKernel(r, h):
    """
    Returns the quantity equivalent to (fraction of mass enclosed)/ r^3 for a cubic-spline mass distribution of compact support radius h. Used to calculate the softened gravitational force.

    Arguments:
    r - radius
    h - softening
    """
    if r > h:
        return 1.0 / (r * r * r)
    hinv = 1.0 / h
    q = r * hinv
    if q <= 0.5:
        return (10.666666666666666666 + q * q * (-38.4 + 32.0 * q)) * hinv * hinv * hinv
    else:
        return (
            (21.333333333333 - 48.0 * q + 38.4 * q * q - 10.666666666667 * q * q * q - 0.066666666667 / (q * q * q))
            * hinv
            * hinv
            * hinv
        )


@njit(fastmath=True, cache=True)
def TidalKernel(r, h):
    """
    Returns the coefficient of dx_i dx_j in the softened tidal tensor of a cubic-spline mass distribution of compact support radius h, equal to -(1/r) d/dr [M(<r)/(M r^3)].

    The tidal tensor of a single element of mass m at separation dx = x_source - x_target is then T_ij = G m (TidalKernel(r,h) dx_i dx_j - ForceKernel(r,h) delta_ij), reducing to the point-mass 3 dx_i dx_j / r^5 - delta_ij / r^3 for r > h. Unlike 3/r^5 it is finite at r=0, approaching 384/(5 h^5).

    Arguments:
    r - radius
    h - softening
    """
    if r > h:
        r2 = r * r
        return 3.0 / (r2 * r2 * r)
    hinv = 1.0 / h
    hinv2 = hinv * hinv
    hinv5 = hinv2 * hinv2 * hinv
    q = r * hinv
    if q <= 0.5:
        return (76.8 - 96.0 * q) * hinv5
    qinv = 1.0 / q
    qinv3 = qinv * qinv * qinv
    return (48.0 * qinv - 76.8 + 32.0 * q - 0.2 * qinv3 * qinv * qinv) * hinv5


@njit(fastmath=True, cache=True)  # ([float64(float64,float64)])
def PotentialKernel(r, h):
    """
    Returns the equivalent of -1/r for a cubic-spline mass distribution of compact support radius h. Used to calculate the softened gravitational potential.

    Arguments:
    r - radius
    h - softening
    """
    if h == 0.0:
        return -1.0 / r
    hinv = 1.0 / h
    q = r * hinv
    if q <= 0.5:
        return (-2.8 + q * q * (5.33333333333333333 + q * q * (6.4 * q - 9.6))) * hinv
    elif q <= 1:
        return (
            -3.2
            + 0.066666666666666666666 / q
            + q * q * (10.666666666666666666666 + q * (-16.0 + q * (9.6 - 2.1333333333333333333333 * q)))
        ) * hinv
    else:
        return -1.0 / r


@njit(fastmath=False, cache=True)
def PotentialKernelSelected(r, h, kernel_id):
    """Unit-mass potential; h is the pair's compact-support radius.

    kernel_id is 0 for the existing cubic spline, 1 for Wendland C2.
    Wendland's normalized density is 21/(2*pi*h**3)*(1-q)**4*(1+4*q).
    Integrating this density gives the polynomial below (q=r/h).
    Reference convention: SWIFT c1c602152b7745c15e0790940069a3d314781dfb.
    """
    if r == 0.0:
        if h == 0.0:
            # Keep the singularity until the Python wrapper can report it,
            # without raising inside a parallel loop. Zero masses are skipped.
            return -float("inf")
        return (-2.8 if kernel_id == 0 else -3.0) / h
    if h == 0.0 or r >= h:
        return -1.0 / r
    if kernel_id == 0:
        return PotentialKernel(r, h)
    q = r / h
    q2 = q * q
    return (-3.0 + q2 * (7.0 + q2 * (-21.0 + q * (28.0 + q * (-15.0 + 3.0 * q))))) / h
