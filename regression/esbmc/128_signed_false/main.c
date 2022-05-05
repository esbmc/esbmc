#include <assert.h>
int main() {
    __int128 X;
    __ESBMC_assume(X>0);
    __int128 Z = 2;
    Z = Z << 70;
    assert((X+Z) > 0);
}