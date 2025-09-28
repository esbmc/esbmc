#include <math.h>
#include <assert.h>

/* Large numbers */

int main()
{
    // Large but finite results
    double result = pow(10.0, 10.0);
    assert(result == 1e10);
    
    // Test for overflow to infinity
    assert(isinf(pow(10.0, 1000.0)));
    
    // Test for underflow to zero
    assert(pow(10.0, -1000.0) == 0.0);
}