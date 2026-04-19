#include <assert.h>
#include <stdint.h>
#include <string.h>

int main() {
    uint32_t value = 0x12345678;
    uint8_t bytes[4];
    
    // Copy bytes maintaining byte order
    memcpy(bytes, &value, sizeof(value));
    
    // Test that we can extract and reconstruct
    uint32_t reconstructed = 0;
    for (int i = 0; i < 4; i++) {
        reconstructed |= ((uint32_t)bytes[i] << (i * 8));
    }
    
    assert(reconstructed == value);
    
    return 0;
}
