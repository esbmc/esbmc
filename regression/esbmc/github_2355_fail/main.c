#include <assert.h>

int nand(int a, int b) {
    return !(a & b);  // NAND is the negation of AND
}

int main() {
    // Test NAND operation
    assert(nand(1, 1) != 0); // 1 NAND 1 should be 0
    return 0;
}
