// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual to yul_pack_tuple_storage_ref_load_pass: tuple-return BalanceLib
// a TUPLE-RETURN library function taking a `P storage` pointer. The precise
// single-slot pack/unpack lowering is correct (see yul_pack_storage_ptr_*),
// but a SEPARATE, PRE-EXISTING frontend bug makes a tuple-return library call
// with a storage-ref parameter read a STALE copy of the caller's state
// variable: the call is queued to a back-block (solidity_convert_tuple.cpp)
// and its storage argument is bound before the preceding `store` call's
// copy-back updates the state var, so `load` sees the pre-store value.
//
// This reproduces WITHOUT any assembly (plain `a = p.a; b = p.b;` tuple load
// fails identically), proving it is orthogonal to the Yul struct-slot lowering.
// The tuple-return named-return binding is now emitted at the fall-through
// exit (solidity_convert_modifier.cpp), so the caller reads the produced
// values. This is the aqua BalanceLib round-trip.
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

    function load(P storage p) internal view returns (uint248 a, uint8 b) {
        assembly {
            let packed := sload(p.slot)
            a := and(packed, 0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff)
            b := shr(248, packed)
        }
    }
}

contract H {
    L.P s;

    function check(uint248 va, uint8 vb) public {
        L.store(s, va, vb);
        (uint248 ra, uint8 rb) = L.load(s);
        assert(ra == va);
        assert(rb == vb + 1);
    }
}
