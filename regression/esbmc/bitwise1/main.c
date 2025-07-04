#include <assert.h>
#include <stdio.h>

void test_xor_operations() {
    // Test case 1: Basic XOR operation
    int a = 5;  // 0101 in binary
    int b = 3;  // 0011 in binary
    int result = a ^ b;  // 0101 ^ 0011 = 0110 (should be 6)
    assert(result == 6);
    printf("Test case 1 passed\n");

    // Test case 2: XOR with zero (should return the same number)
    int c = 0;  // 0000 in binary
    result = a ^ c;  // 0101 ^ 0000 = 0101 (should be 5)
    assert(result == 5);
    printf("Test case 2 passed\n");

    // Test case 3: XOR of a number with itself (should return 0)
    result = a ^ a;  // 0101 ^ 0101 = 0000 (should be 0)
    assert(result == 0);
    printf("Test case 3 passed\n");

    // Test case 4: XOR with all bits set (should invert the bits)
    int d = ~0;  // All bits set (typically -1)
    result = a ^ d;  // 0101 ^ 1111...1111 = 1010...1010 (~5)
    assert(result == ~5);
    printf("Test case 4 passed\n");
    
    // Test case 5: XOR with alternating bits
    int e = 0x55;  // 01010101 in binary
    int f = 0xAA;  // 10101010 in binary
    result = e ^ f;  // 01010101 ^ 10101010 = 11111111 (should be 255)
    assert(result == 255);
    printf("Test case 5 passed\n");
}

int main() {
    // Run all the test cases
    test_xor_operations();
    return 0;
}


