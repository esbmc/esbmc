// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Library `P storage` pointer (the 1inch/aqua BalanceLib storage-ref pattern):
// single-slot pack via `sstore(p.slot, ...)`, unpack via `sload(p.slot)`. A
// single-value `view` load isolates the precise struct-slot lowering + the
// storage-ref copy-back from the (separate, pre-existing) tuple-return
// staleness bug pinned in yul_pack_tuple_storage_ref_load_knownbug.
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
        assert(rb == vb);
    }
}
