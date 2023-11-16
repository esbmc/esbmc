#include <math.h>

float sinf(float x)
{
__ESBMC_HIDE:;
    /**
        Highly imprecise implementation of sinf(x) on floating-point operands
        which satisfies the following properties:
        sinf(x) >= -1
        sinf(x) <= 1
        sinf(x) = -sinf(-x)
        dsinf(x)/dx = cosf(x)
     */

    return cosf(x - M_PI_2);
}
