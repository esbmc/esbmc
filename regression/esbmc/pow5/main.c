#include <math.h>
#include <assert.h>

/* Basic negative exponents */

int main()
{
    assert(pow(2.0, -1.0) == 0.5);
    assert(pow(2.0, -2.0) == 0.25);
    assert(pow(4.0, -0.5) == 0.5);
    assert(pow(10.0, -1.0) == 0.1);
}