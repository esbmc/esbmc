#include <math.h>

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
    if(x <= -1.0f)
        return -1.0f;
    if(x >= 1.0f)
        return 1.0f;
    return x;
}
