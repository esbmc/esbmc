#include <assert.h>

int my_nand(int a, int b) {
    return !(a & b);  // NAND is the negation of AND
}

int main() {
    // Test NAND operation
    assert(my_nand(1, 1) == 0); // 1 NAND 1 should be 0
    assert(my_nand(1, 0) == 1); // 1 NAND 0 should be 1
    assert(my_nand(0, 1) == 1); // 0 NAND 1 should be 1
    assert(my_nand(0, 0) == 1); // 0 NAND 0 should be 1
    return 0;
}

