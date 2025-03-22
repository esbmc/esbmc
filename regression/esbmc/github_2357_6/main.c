#include <assert.h>
#include <limits.h>
#include <stdint.h>

int main() {
    // Signed division leading to overflow (INT_MIN / -1)
    int min_signed = INT_MIN;
    int div_overflow = min_signed / -1; // UB, would overflow
    return 0;
}

