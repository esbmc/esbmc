#include <math.h>

#define ALMOST_ONE 0.9999999f // necessary to ensure expm1f(x) >= x for x <= 0

float expf(float x)
{
__ESBMC_HIDE:;
    
	// unimplemented operational models
    // should return VERIFICATION UNKNOWN
    __ESBMC_unreachable();

    return 0.0f;
}

float expm1f(float x)
{
__ESBMC_HIDE:;

    /**
        Highly imprecise implementation of expm1f(x) on floating-point operands
        which satisfies the following properties:
        expm1f(x) >= -1
        expm1f(x) >= x
        expm1f(-Inf) = -1
        expm1f(0) = 0
        expm1f(+Inf) = Inf
        dexpm1f(x=0)/dx = 1
     */

    // handle positive infinity
    if(x >= 1.0f)
        return 1.0f / 0.0f;

    // first-order Pade' approximant
    return 1.0f / (1.0f - x) - ALMOST_ONE;
}
