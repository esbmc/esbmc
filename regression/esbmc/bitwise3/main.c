#include <assert.h>
#include <stdio.h>

// Function to perform boolean NXOR (XNOR)
int bool_nxor(int a, int b) {
    return !(a ^ b);
}

// Function to perform bitwise NXOR (XNOR) on integers
int int_nxor(int a, int b) {
    return ~(a ^ b);
}

void test_nxor() {
    // Boolean NXOR test cases
    assert(bool_nxor(0, 0) == 1);
    assert(bool_nxor(0, 1) == 0);
    assert(bool_nxor(1, 0) == 0);
    assert(bool_nxor(1, 1) == 1);

    // Integer NXOR test cases
    assert(int_nxor(0b1100, 0b1010) == ~(0b1100 ^ 0b1010));
    assert(int_nxor(0xF0F0, 0x0F0F) == ~(0xF0F0 ^ 0x0F0F));
    assert(int_nxor(123, 456) == ~(123 ^ 456));
    assert(int_nxor(-5, 3) == ~(-5 ^ 3));
    assert(int_nxor(0, 0) == ~0);
    
    printf("All NXOR tests passed!\n");
}

int main() {
    test_nxor();
    return 0;
}
