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
        tanhf(Inf) = 1
        tanhf(x) = -tanhf(-x)
        dtanhf(x)/dx <= 1
     */

    // handle non-finite numbers
    if(isnan(x))
        return x;
    if(isinf(x))
        return (x > 0)? 1.0f: -1.0f;

    // softsign approximation
    if(x < 0)
        return x / (1.0f - x);
    else
        return x / (1.0f + x);
}
