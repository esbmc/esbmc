// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Distinct packing shape from the 248/8 split: `P{uint128 a; uint128 b}` packs
// a at [0,128), b at [128,256). Pins that the precise lowering derives field
// bit-offsets GENERICALLY from the struct layout (128 here, not a hard-coded
// 248) and that neither field bleeds into the other. Under the legacy havoc
// fallback the round-trip FAILs. KNOWNBUG until struct-slot lowering lands.
contract H {
    struct P {
        uint128 a;
        uint128 b;
    }
    P s;

    function check(uint128 va, uint128 vb) public {
        assembly {
            let packed := or(va, shl(128, vb))
            sstore(s.slot, packed)
        }
        uint128 ra;
        uint128 rb;
        assembly {
            let packed := sload(s.slot)
            ra := and(packed, 0xffffffffffffffffffffffffffffffff)
            rb := shr(128, packed)
        }
        assert(ra == va);
        assert(rb == vb);
    }
}
