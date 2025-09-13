#include <math.h>
#include <assert.h>

int main()
{
    int a = 2;
    int b = 3;
    assert(pow(a, b) == 9); // pow(2,3) == 8 not 9; should fail
}