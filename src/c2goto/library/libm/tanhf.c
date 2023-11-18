#include <math.h>

#define ALMOST_ONE 0.999999f // necessary to ensure dtanhf(x)/dx <= 1 and dlogistic(x)/dx <= 0.25

float tanhf(float x)
{
__ESBMC_HIDE:;

    /**
        Highly imprecise implementation of tanhf(x) on floating-point operands
        which satisfies the following properties:
        tanhf(x) >= -1
        tanhf(x) <= 1
        tanhf(-Inf) = -1
        tanhf(0) = 0
        tanhf(Inf) = 1
        tanhf(x) = -tanhf(-x)
        dtanhf(x)/dx <= 1
     */

    // handle NaNs
    if(isnan(x))
        return x;

    // piece-wise linear approximation
    x *= ALMOST_ONE;
    if(x <= -1.0f)
        return -1.0f;
    if(x >= 1.0f)
        return 1.0f;
    return x;
}
