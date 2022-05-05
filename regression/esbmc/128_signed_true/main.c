#include <assert.h>
int main() {
    __int128 Z = 2;
    Z = Z << 70;
    assert(Z > 0);
}