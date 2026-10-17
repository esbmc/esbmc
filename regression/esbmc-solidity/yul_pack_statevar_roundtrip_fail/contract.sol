// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual to yul_pack_statevar_roundtrip_pass: same single-slot pack/unpack,
// but a wrong assertion. After precise lowering ra == va, so `ra == va + 1` is
// violable (universal claim fails). This stays FAILED before AND after the fix
// — it guards against the pass test becoming vacuous.
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
        assert(ra == va + 1);
    }
}
