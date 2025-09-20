#include <math.h>
#include <assert.h>

/* Test basic integer powers */

int main()
{
    assert(pow(2.0, 3.0) == 8.0);
    assert(pow(3.0, 2.0) == 9.0);
    assert(pow(5.0, 0.0) == 1.0);
    assert(pow(1.0, 100.0) == 1.0);
    
    assert(pow(-2.0, 3.0) == -8.0);
    assert(pow(-2.0, 2.0) == 4.0);
    assert(pow(-3.0, 4.0) == 81.0);
}