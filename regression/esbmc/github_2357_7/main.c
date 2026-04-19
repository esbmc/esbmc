#include <assert.h>
#include <limits.h>
#include <stdint.h>

int main() {
    // Casting a large unsigned number to a small signed type
    uint32_t large_uint = UINT_MAX;
    int8_t small_signed = (int8_t) large_uint;
    assert(small_signed == -1); // Wraparound

    return 0;
}

