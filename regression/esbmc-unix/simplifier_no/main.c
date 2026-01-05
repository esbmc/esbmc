#include <assert.h>
#include <stdint.h>

extern void __VERIFIER_assume(int);

// Test 1: Explicit storage with wraparound
void test_explicit_wrap_8bit() {
    unsigned char x = 10;
    unsigned char y = 12;
    
    unsigned char prod_x = x * 128;
    unsigned char prod_y = y * 128;
    
    assert((prod_x == prod_y) == (x == y));
}

// Test 2: uint32_t wraparound with c=2
void test_uint32_even_constant() {
    uint32_t x = 0;
    uint32_t y = 2147483648U;
    
    assert(((x * 2U) == (y * 2U)) == (x == y));
}

// Test 3: uint32_t with larger constant
void test_uint32_large_constant() {
    uint32_t x = 1;
    uint32_t y = 2147483649U;
    uint32_t c = 2147483648U;
    
    assert(((x * c) == (y * c)) == (x == y));
}

// Test 4: uint64_t wraparound
void test_uint64_wraparound() {
    uint64_t x = 0;
    uint64_t y = 9223372036854775808ULL;
    
    assert(((x * 2ULL) == (y * 2ULL)) == (x == y));
}

// Test 5: uint32_t with smaller difference
void test_uint32_smaller_diff() {
    uint32_t x = 100;
    uint32_t y = 2147483748U;
    
    assert(((x * 2U) == (y * 2U)) == (x == y));
}

// Test 6: Symbolic test with uint32_t
void test_uint32_symbolic() {
    uint32_t x, y;
    
    __VERIFIER_assume((x * 2U) == (y * 2U));
    __VERIFIER_assume(x != y);
    
    assert(0);
}

// Test 7: Cast to smaller type in comparison
void test_cast_to_smaller() {
    int x = 10;
    int y = 12;
    
    assert(((unsigned char)(x * 128) == (unsigned char)(y * 128)) == (x == y));
}

// Test 8: uint16_t wraparound
void test_uint16_wraparound() {
    uint16_t x = 0;
    uint16_t y = 32768;
    
    assert(((x * 2) == (y * 2)) == (x == y));
}

// Test 9: Multiple of 4 with uint32_t
void test_uint32_multiple_of_4() {
    uint32_t x = 0;
    uint32_t y = 1073741824U;
    
    assert(((x * 4U) == (y * 4U)) == (x == y));
}

// Test 10: Subtraction with constants
void test_subtraction_constants() {
    int d = 100;
    int c1 = 10;
    int c2 = 20;
    
    assert(((d - c1) == (d - c2)) == (c1 == c2));
}

// Test 11: Subtraction symbolic with overflow
void test_subtraction_symbolic_overflow() {
    int d, c, e;
    
    __VERIFIER_assume(d == 2147483647);
    __VERIFIER_assume(c == -1);
    __VERIFIER_assume(e == -1);
    
    int left = d - c;
    int right = d - e;
    
    assert(left == right);
}

// Test 12: Negation with mixed types
void test_negation_different_types() {
    signed char x = 5;
    signed short y = 5;
    
    assert(((-x) == (-y)) == (x == y));
}

// Test 13: Bitwise NOT with mixed types
void test_bitnot_different_types() {
    unsigned char x = 5;
    unsigned short y = 5;
    
    assert(((~x) == (~y)) == (x == y));
}

// Test 14: Multiplication with odd constant (should work)
void test_multiplication_odd_constant() {
    unsigned char x = 10;
    unsigned char y = 12;
    
    assert(((x * 5) == (y * 5)) == (x == y));
}

// Test 15: uint32_t symbolic with even constant
void test_uint32_symbolic_even() {
    uint32_t x, y;
    
    __VERIFIER_assume((x * 128U) == (y * 128U));
    __VERIFIER_assume(x != y);
    
    assert(0);
}

int main() {
    test_explicit_wrap_8bit();
    test_uint32_even_constant();
    test_uint32_large_constant();
    test_uint64_wraparound();
    test_uint32_smaller_diff();
    test_uint32_symbolic();
    test_cast_to_smaller();
    test_uint16_wraparound();
    test_uint32_multiple_of_4();
    test_subtraction_constants();
    test_subtraction_symbolic_overflow();
    test_negation_different_types();
    test_bitnot_different_types();
    test_multiplication_odd_constant();
    test_uint32_symbolic_even();
    
    return 0;
}
