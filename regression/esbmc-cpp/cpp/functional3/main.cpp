#include <cassert>
#include <functional>  // for bitwise operations

int main() {
    // Test bit_and
    std::bit_and<int> bit_and_op;

    // 5 (0101) & 3 (0011) = 1 (0001)
    assert(bit_and_op(5, 3) == 1);

    // 8 (1000) & 4 (0100) = 0 (0000)
    assert(bit_and_op(8, 4) == 0);

    // 7 (0111) & 5 (0101) = 5 (0101)
    assert(bit_and_op(7, 5) == 5);

    // Test bit_or
    std::bit_or<int> bit_or_op;

    // 5 (0101) | 3 (0011) = 7 (0111)
    assert(bit_or_op(5, 3) == 7);

    // 8 (1000) | 4 (0100) = 12 (1100)
    assert(bit_or_op(8, 4) == 12);

    // 7 (0111) | 5 (0101) = 7 (0111)
    assert(bit_or_op(7, 5) == 7);

    // Test bit_xor
    std::bit_xor<int> bit_xor_op;

    // 5 (0101) ^ 3 (0011) = 6 (0110)
    assert(bit_xor_op(5, 3) == 6);

    // 8 (1000) ^ 4 (0100) = 12 (1100)
    assert(bit_xor_op(8, 4) == 12);

    // 7 (0111) ^ 5 (0101) = 2 (0010)
    assert(bit_xor_op(7, 5) == 2);

    // Test bit_not
    std::bit_not<int> bit_not_op;

    // ~5 (0101) = -6 (in two's complement representation, 11111111111111111111111111111010)
    assert(bit_not_op(5) == -6);

    // ~8 (1000) = -9 (in two's complement representation, 11111111111111111111111111110111)
    assert(bit_not_op(8) == -9);

    // ~0 (0000) = -1 (in two's complement representation, 11111111111111111111111111111111)
    assert(bit_not_op(0) == -1);

    return 0;
}

