#include <math.h>
#include <assert.h>

int main()
{
    assert(pow(-4.0, 0.5) == 2); // Should return NaN not 2
}