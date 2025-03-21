#include <assert.h>
#include <limits.h>
#include <stdint.h>

int main() {
    unsigned int u_x = UINT_MAX;
    u_x += 1;
    assert(u_x == 0);
    return 0;
}


