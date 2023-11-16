#include <math.h>

float expf(float x)
{
__ESBMC_HIDE:;
    /**
        Highly imprecise implementation of expf(x) on floating-point operands
        which satisfies the following properties:
        expf(x) >= 0
        expf(x) >= 1 + x
        expf(-Inf) = 0
        expf(0) = 1
        expf(+Inf) = Inf
        dexpf(x=0)/dx = 1
        logf(expf(x)) = x
     */

    // handle NaN inputs
    if(isnan(x))
        return x;

    // handle infinities (x*x will overflow before x)
    float xx = x * x;
    if(isinf(xx)) {
        if(x < 0.0f)
            return 0.0f; // expf(-Inf) = 0
        else
            return 1.0f / 0.0f; // expf(+Inf) = Inf
    }

    // inverse function of the logf approximation
    // i.e. inverse((x - 1) / sqrt(x))
    // it is numerically unstable for x < 0
    // since it relies on x - sqrt(x^2 + 4) cancelling out
    return 0.5 * x * (x + sqrtf(xx + 4.0f)) + 1.0f;
}

float expm1f(float x)
{
__ESBMC_HIDE:;
    return expf(x) - 1.0f;
}
