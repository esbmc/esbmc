#include <math.h>
#include <assert.h>

/* Infinity cases */

int main()
{
    // Positive infinity exponent
    assert(isinf(pow(2.0, INFINITY)));
    assert(pow(0.5, INFINITY) == 0.0);
    assert(pow(-1.0, INFINITY) == 1.0);  // Special case for -1
    
    // Negative infinity exponent
    assert(pow(2.0, -INFINITY) == 0.0);
    assert(isinf(pow(0.5, -INFINITY)));
    
    // Infinite base
    assert(isinf(pow(INFINITY, 2.0)));
    assert(pow(INFINITY, -1.0) == 0.0);
    assert(isinf(pow(-INFINITY, 3.0)));
    assert(isinf(pow(-INFINITY, 2.0)));
}