#include <assert.h>
#include <limits.h>
#include <stdint.h>

int main() {
    // Typecasting signed negative to unsigned
    int negative_value = -1;
    unsigned int casted_unsigned = (unsigned int) negative_value;
    assert(casted_unsigned == UINT_MAX); // Should become max unsigned value
    return 0;
}

