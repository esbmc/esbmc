#include <assert.h>

int test_xor_zero_vs_false() {
    int x = nondet_int();
    int result1 = x ^ 0;  // Integer 0
    assert(result1 == x);
    int zero = 0;
    int result2 = x ^ zero;
    assert(result2 == x);
    return 0;
}

int test_unsigned_comparison() {
    unsigned int u = nondet_uint();
    int result1 = (u < 0);  // Integer 0, not boolean false
    assert(result1 == 0);
    int zero = 0;
    int result2 = (u < zero);
    assert(result2 == 0);
    
    return 0;
}

int test_wide_bitwidth() {
    unsigned char c = nondet_uchar();
    unsigned short s = nondet_ushort();
    unsigned long long ll = nondet_ullong();
    assert((c & (unsigned char)(-1)) == c);
    assert((s & (unsigned short)(-1)) == s);
    assert((ll & (unsigned long long)(-1)) == ll);
    return 0;
}

int test_vector_simplifications() {
    typedef unsigned v4 __attribute__((vector_size(16)));
    v4 x = {1, 2, 3, 4};
    v4 zero = {0, 0, 0, 0};
    v4 result1 = x ^ zero;
    assert(result1[0] == 1);
    assert(result1[1] == 2);
    v4 result2 = x ^ x;
    assert(result2[0] == 0);
    assert(result2[1] == 0);
    return 0;
}

int test_mixed_operations() {
    int x = nondet_int();
    _Bool b = 0;
    int i = 0;
    int result1 = x ^ b;
    int result2 = x ^ i;
    assert(result2 == x);
    return 0;
}

int main() {
    test_xor_zero_vs_false();
    test_unsigned_comparison();
    test_wide_bitwidth();
    test_vector_simplifications();
    test_mixed_operations();
    return 0;
}
