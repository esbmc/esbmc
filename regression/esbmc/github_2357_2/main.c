#include <assert.h>
#include <limits.h>

int main() {
    // Unsigned integer overflow test
    unsigned int u_x = UINT_MAX; // Maximum value of unsigned int
    u_x += 1;  // Overflow occurs here

    assert(u_x == 0); // Expected: wrapping around to 0

    // Signed integer overflow
    int s_x = INT_MAX;
    s_x += 1;  // This causes undefined behavior

    return 0;
}

