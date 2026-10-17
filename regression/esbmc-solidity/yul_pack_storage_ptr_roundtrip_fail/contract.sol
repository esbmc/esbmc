// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual to yul_pack_storage_ptr_roundtrip_pass: single-value library
// storage-ref round-trip with a wrong assertion (`rb == vb + 1`), violable
// after precise lowering. Stays FAILED before and after — guards non-vacuity.
library L {
    struct P {
        uint248 a;
        uint8 b;
    }

    function store(P storage p, uint248 a, uint8 b) internal {
        assembly {
            let packed := or(a, shl(248, b))
            sstore(p.slot, packed)
        }
    }

    function loadB(P storage p) internal view returns (uint8 b) {
        assembly {
            b := shr(248, sload(p.slot))
        }
    }
}

contract H {
    L.P s;

    function check(uint248 va, uint8 vb) public {
        L.store(s, va, vb);
        uint8 rb = L.loadB(s);
        assert(rb == vb + 1);
    }
}
