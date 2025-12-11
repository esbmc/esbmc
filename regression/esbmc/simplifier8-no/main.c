#include <assert.h>
#include <stdint.h>

extern uint8_t nondet_uint8_t();

int main() {
    uint8_t a = nondet_uint8_t();
    uint8_t b = nondet_uint8_t();
    uint8_t c = nondet_uint8_t(); // for nested checks

    // identity: (a & ~b) | (a ^ b) == (a ^ b)
    assert( (((a & ~b) | (a ^ b)) ) == ( (a ^ b) ) );

    // identity: (~a ^ b) | (a & b) == (~a ^ b)
    assert( ((((~a) ^ b) | (a & b)) ) == ( ((~a) ^ b) ) );

    // identity: (~a & b) | ~(a | b) == ~a
    assert( ((((~a) & b) | (~(a | b))) ) == ( (~a) ) );

    // identity: (a ^ b) | (a | b) == (a | b)
    assert( (((a ^ b) | (a | b)) ) == ( (a | b) ) );

    // Nested checks (replace LHS by RHS and combine with c)
    assert( ( (((a & ~b) | (a ^ b)) | c) ) == ( ((a ^ b) | c) ) );
    assert( ( (((a & ~b) | (a ^ b)) & c) ) == ( ((a ^ b) & c) ) );

    return 0;
}


