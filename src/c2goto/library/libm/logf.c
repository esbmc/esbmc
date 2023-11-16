#include <math.h>

float logf(float x)
{
__ESBMC_HIDE:;
    /**
        Highly imprecise implementation of logf(x) on floating-point operands
        which satisfies the following properties:
        logf(x) <= x - 1
        logf(0) = -Inf
        logf(1) = 0
        logf(Inf) = Inf
        dlogf(x=1)/dx = 1
        expf(logf(x)) = x
     */

    // handle negative inputs
    if(x < 0.0f)
        return 0.0f / 0.0f; // return NaN

    // handle positive infinity
    if(isinf(x))
        return x;

    // ad-hoc fractional Pade' approximant
    return (x - 1.0f) / sqrtf(x);
}

float log1pf(float x)
{
__ESBMC_HIDE:;
    return logf(x + 1.0f);
}
