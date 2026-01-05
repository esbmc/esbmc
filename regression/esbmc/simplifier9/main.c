#include <assert.h>
#include <stdint.h>

extern uint16_t nondet_uint16_t();

int main() {
    uint16_t a = nondet_uint16_t();
    uint16_t b = nondet_uint16_t();
    uint16_t c = nondet_uint16_t(); // for nested checks

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


