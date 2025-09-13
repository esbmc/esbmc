#include <math.h>
#include <assert.h>

/* NaN cases */

int main()
{
    // NaN propagation
    assert(isnan(pow(NAN, 2.0)));
    assert(isnan(pow(2.0, NAN)));
    assert(isnan(pow(NAN, NAN)));
    
    // Negative base with non-integer exponent should return NaN
    assert(isnan(pow(-2.0, 0.5)));
    assert(isnan(pow(-1.5, 2.5)));
}