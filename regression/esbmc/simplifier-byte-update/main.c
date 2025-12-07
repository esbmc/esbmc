#include <stdint.h>
#include <assert.h>

#define N 5

void demo_performance_impact(void)
{
    uint32_t buffer[N];
    uint8_t *bytes = (uint8_t *)buffer;

    // WITHOUT simplification, this creates N byte_update operations:
    for (int i = 0; i < N; i++) {
        bytes[i] = i;  // N × byte_update
    }
    // And this creates N byte_extract operations chained with updates:
    for (int i = 0; i < N; i++) {
        uint8_t val = bytes[i];  // N × byte_extract(byte_update(...))
        assert(val == i);
    }
}

int main(void)
{
    demo_performance_impact();
    return 0;
}
