#include <math.h>
#include <assert.h>

/* Test special cases */

int main()
{
    // Identity cases
    assert(pow(1.0, 0.0) == 1.0);
    assert(pow(1.0, 1.0) == 1.0);
    assert(pow(1.0, -1.0) == 1.0);
    assert(pow(0.0, 0.0) == 1.0);  // Mathematically undefined, but C standard defines as 1
    
    // Zero base cases
    assert(pow(0.0, 1.0) == 0.0);
    assert(pow(0.0, 2.0) == 0.0);
    
    // Zero exponent
    assert(pow(5.0, 0.0) == 1.0);
    assert(pow(-5.0, 0.0) == 1.0);
    assert(pow(0.0, 0.0) == 1.0);
}