#include <math.h>
#include <assert.h>

/* Test powers of 2 */

int main()
{
    // Powers of 2
    assert(pow(2.0, 1.0) == 2.0);
    assert(pow(2.0, 2.0) == 4.0);
    assert(pow(2.0, 3.0) == 8.0);
    assert(pow(2.0, 10.0) == 1024.0);
    
    // Negative powers of 2
    assert(pow(2.0, -1.0) == 0.5);
    assert(pow(2.0, -2.0) == 0.25);
    
    // Other powers of 2 (0.5, 4.0, 8.0, etc.)
    assert(pow(0.5, 1.0) == 0.5);
    assert(pow(0.5, 2.0) == 0.25);
    assert(pow(4.0, 2.0) == 16.0);
}