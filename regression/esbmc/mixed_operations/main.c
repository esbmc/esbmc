#include <assert.h>
#include <stdint.h>

int main() {
    uint32_t original = 0x12345678;
    uint8_t *bytes = (uint8_t*)&original;
    
    // Extract individual bytes
    uint8_t b0 = bytes[0];
    uint8_t b1 = bytes[1];
    uint8_t b2 = bytes[2];
    uint8_t b3 = bytes[3];
    
    // Reconstruct using concatenation (reverse byte order for big-endian result)
    uint32_t reconstructed = ((uint32_t)b3 << 24) | 
                            ((uint32_t)b2 << 16) | 
                            ((uint32_t)b1 << 8) | 
                            b0;
    
    // Should equal original value
    assert(reconstructed == original);
    
    return 0;
}
