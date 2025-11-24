#include <assert.h>
#include <stdint.h>

int main() {
    uint32_t val = 0xCAFEBABE;
    uint8_t b0 = (val >> 0) & 0xFF;
    uint8_t b1 = (val >> 8) & 0xFF;
    uint8_t b2 = (val >> 16) & 0xFF;
    uint8_t b3 = (val >> 24) & 0xFF;

    assert(b0 == 0xBE);
    assert(b1 == 0xBA);
    assert(b2 == 0xFE);
    assert(b3 == 0xCA);

    // Rebuild from bytes
    uint32_t rebuild = ((uint32_t)b3 << 24) |
                       ((uint32_t)b2 << 16) |
                       ((uint32_t)b1 << 8) |
                       b0;

    assert(rebuild == val);

    return 0;
}

