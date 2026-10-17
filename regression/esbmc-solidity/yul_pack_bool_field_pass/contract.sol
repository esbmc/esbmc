// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Bool field in a single slot: `P{bool b; uint248 x}` — b at byte 0 (1 byte),
// x at bits [8,256). Pins that the precise lowering treats bool as a 1-byte
// field (offset for x is 8, not 0) and round-trips 0/1 semantics rather than a
// raw masked integer. KNOWNBUG until struct-slot lowering lands.
contract H {
    struct P {
        bool b;
        uint248 x;
    }
    P s;

    function check(bool vb, uint248 vx) public {
        assembly {
            let packed := or(and(vb, 0x1), shl(8, vx))
            sstore(s.slot, packed)
        }
        bool rb;
        uint248 rx;
        assembly {
            let packed := sload(s.slot)
            rb := and(packed, 0x1)
            rx := shr(8, packed)
        }
        assert(rb == vb);
        assert(rx == vx);
    }
}
