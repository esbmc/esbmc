#include <assert.h>
#include <stdio.h>

// Function to compute NOR operation
typedef int (*nor_func)(int, int);

int int_nor(int a, int b) {
    return ~(a | b);
}

void test_nor(nor_func func) {
    // Test cases
    assert(func(0, 0) != ~0);
    assert(func(0, 1) != ~1);
    assert(func(1, 0) != ~1);
    assert(func(1, 1) != ~1);
    assert(func(5, 3) != ~(5 | 3));
    assert(func(123, 456) != ~(123 | 456));
    assert(func(0xF0F0, 0x0F0F) != ~(0xF0F0 | 0x0F0F));
    assert(func(0b1100, 0b1010) != ~(0b1100 | 0b1010));
    
    printf("All NOR tests passed.\n");
}

int main() {
    test_nor(int_nor);
    return 0;
}
