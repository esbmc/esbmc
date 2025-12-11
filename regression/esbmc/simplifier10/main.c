
#include <assert.h>
#include <stdint.h>

// Auto-generated regression tests for bitwise identities
// Width: 32 bits, C type: uint32_t

extern uint32_t nondet_uint32_t();

int main() {
    uint32_t a = nondet_uint32_t();
    uint32_t b = nondet_uint32_t();
    uint32_t c = nondet_uint32_t(); // for nested checks

    // identity: (a & ~b) | (a ^ b) == (a ^ b)
    assert( (((a & ~b) | (a ^ b)) ) == ( (a ^ b) ) );

    // identity: (~a ^ b) | (a & b) == (~a ^ b)
    assert( ((((~a) ^ b) | (a & b)) ) == ( ((~a) ^ b) ) );

    // identity: (~a | b) | (a ^ b) == -1 (all bits set)  (all bits set)
    assert( ((((~a) | b) | (a ^ b)) ) == (((uint32_t)~( (uint32_t)0 )) /* all ones for uint32_t */) );

    // identity: (~a & b) | ~(a | b) == ~a
    assert( ((((~a) & b) | (~(a | b))) ) == ( (~a) ) );

    // identity: ~(a ^ b) | (a | b) == -1 (all bits set)  (all bits set)
    assert( (((~(a ^ b)) | (a | b)) ) == (((uint32_t)~( (uint32_t)0 )) /* all ones for uint32_t */) );

    // identity: (a ^ b) | (a | b) == (a | b)
    assert( (((a ^ b) | (a | b)) ) == ( (a | b) ) );

    // Nested checks (replace LHS by RHS and combine with c)
    assert( ( (((a & ~b) | (a ^ b)) | c) ) == ( ((a ^ b) | c) ) );
    assert( ( (((a & ~b) | (a ^ b)) & c) ) == ( ((a ^ b) & c) ) );

    return 0;
}

