#include <assert.h>
#include <stdint.h>

int main() {
    uint16_t high = 0x1234;
    uint16_t low = 0x5678;
    
    // Concatenate to form 32-bit value
    uint32_t result = ((uint32_t)high << 16) | low;
    
    // Expected result: 0x12345678
    assert(result == 0x12345678);
    
    // Test with 8-bit values
    uint8_t byte1 = 0xAB;
    uint8_t byte2 = 0xCD;
    uint16_t concat16 = ((uint16_t)byte1 << 8) | byte2;
    
    assert(concat16 == 0xABCD);
    
    return 0;
}
