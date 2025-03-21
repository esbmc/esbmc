#include <assert.h>
#include <limits.h>
#include <stdint.h>

int main() {
    // Signed integer overflow
    int s_x = INT_MAX;
     s_x += 1; // this leads to UB
    return 0;
}

