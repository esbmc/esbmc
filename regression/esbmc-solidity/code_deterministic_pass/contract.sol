// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// KNOWNBUG: per-address `.code.length` determinism.
//
// The new `_ESBMC_code_of(addr)` helper makes the underlying `.code`
// summary stable per-address, but the `.length` member access on a
// uint256-modeled bytes value falls back to a fresh `nondet_uint` in
// solidity_convert_ref.cpp around line 722-725:
//
//   if (base.type().is_unsignedbv() || base.type().is_signedbv())
//     get_nondet_expr(uint_type(), new_expr);  // fresh nondet
//
// Two reads of `.code.length` therefore disagree even though the
// underlying `.code` value would agree.  Fixing this requires
// threading the original address expression through to a parallel
// `_ESBMC_code_length_of(addr)` helper — orthogonal to the codehash
// fix and tracked separately.
contract C {
    function check(address a) public view {
        uint l1 = a.code.length;
        uint l2 = a.code.length;
        assert(l1 == l2);
    }
}
