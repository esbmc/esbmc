#include <math.h>

float cosf(float x)
{
__ESBMC_HIDE:;
    /**
        Highly imprecise implementation of cosf(x) on floating-point operands
        which satisfies the following properties:
        cosf(x) >= -1
        cosf(x) <= 1
        cosf(x) = cosf(-x)
        dcosf(x)/dx = -sinf(x)
     */

    // handle non-finite numbers
    // TODO: return signaling NaN when x=NaN
    if(isnan(x) || isinf(x))
        return x-x; // return (quiet) NaN

    // restrict x so that 0 < x < M_PI
    if(x < 0) x = -x;
    x = fmodf(x + M_PI, 2.0f * M_PI) - M_PI;

    // piece-wise linear approximation
    if(x <= 0.25 * M_PI)
        return 1.0f;
    if(x <= 0.75 * M_PI) {
        return -x * 4.0f / M_PI + 2.0f;
    }
    return -1.0f;
}
