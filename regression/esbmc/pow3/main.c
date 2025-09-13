#include <math.h>
#include <assert.h>

/* Test fractional powers */

int main()
{
    // Square roots
    assert(pow(4.0, 0.5) == 2.0);
    assert(pow(9.0, 0.5) == 3.0);
    assert(pow(16.0, 0.5) == 4.0);
    
    // Cube roots
    assert(pow(8.0, 1.0/3.0) == 2.0);
    assert(pow(27.0, 1.0/3.0) == 3.0);
}