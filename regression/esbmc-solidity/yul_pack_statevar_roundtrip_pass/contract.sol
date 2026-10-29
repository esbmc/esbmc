// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Single-slot struct packing via inline assembly on a STATE-VARIABLE struct.
// `P{uint248 a; uint8 b}` packs into one 256-bit slot: a at bits [0,248),
// b at bits [248,256). Under the legacy havoc fallback, sstore/sload on a
// non-scalar (struct) `.slot` re-nondets the struct, so ra/rb are nondet and
// the round-trip assertions FAIL. With precise single-slot pack/unpack
// lowering they hold. KNOWNBUG until the T2.4 struct-slot lowering lands.
contract H {
    struct P {
        uint248 a;
        uint8 b;
    }
    P s;

    function check(uint248 va, uint8 vb) public {
        assembly {
            let packed := or(va, shl(248, vb))
            sstore(s.slot, packed)
        }
        uint248 ra;
        uint8 rb;
        assembly {
            let packed := sload(s.slot)
            ra := and(packed, 0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff)
            rb := shr(248, packed)
        }
        assert(ra == va);
        assert(rb == vb);
    }
}
