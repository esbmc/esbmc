#include <assert.h>
#include <stdint.h>

int main() {
    uint32_t value = 0x12345678;
    uint8_t *bytes = (uint8_t*)&value;
    
    // Test little-endian byte extraction
    uint8_t byte0 = bytes[0];  // Should be 0x78
    uint8_t byte1 = bytes[1];  // Should be 0x56
    uint8_t byte2 = bytes[2];  // Should be 0x34
    uint8_t byte3 = bytes[3];  // Should be 0x12
    
    // Verify results
    assert(byte0 == 0x78);
    assert(byte1 == 0x56);
    assert(byte2 == 0x34);
    assert(byte3 == 0x12);
    
    return 0;
}

