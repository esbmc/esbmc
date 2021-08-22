#include <stdlib.h>
#include <assert.h>

int main()
{
    unsigned int x = 7;
    int clz = __builtin_clz(x);

    assert(clz == 29);

    return 0;
}