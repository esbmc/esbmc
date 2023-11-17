#include <math.h>

#define TWO_SQRT_PI 1.1283791670955125738961589031215451716881012586579977136881714434f // TWO_SQRT_PI = 2 / sqrt(pi)

float erff(float x)
{
__ESBMC_HIDE:;

    /**
        Highly imprecise implementation of erff(x) on floating-point operands
        which satisfies the following properties:
        erff(x) >= -1
        erff(x) <= 1
        erff(-Inf) = -1
        erff(0) = 0
        erff(Inf) = 1
        erff(x) = -erff(-x)
        derff(x)/dx <= 2 / sqrt(pi)
     */

    // handle NaNs
    if(isnan(x))
        return x;

    // piece-wise linear approximation
    x *= TWO_SQRT_PI;
    if(x <= -1.0f)
        return -1.0f;
    if(x >= 1.0f)
        return 1.0f;
    return x;
}
