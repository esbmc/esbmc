// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// P1 regression-lock: `S[] arr; arr.push();` (no-arg push of struct
// element) currently CRASHES the symex (core dump).
//
// Root cause: `solidity_convert_ref.cpp:858` emits `gen_zero(elem_type)`
// where `elem_type` is `symbol_typet("tag-S")` for a user struct.
// `util/expr_util.cpp:90` has no `symbol` case, so `gen_zero` returns a
// `nil` exprt; lowering produces `ASSIGN arr[idx]=nil` and symex aborts
// on the nil rhs.
//
// KNOWNBUG until S1+S2 of the push/pop spec-conformance plan land
// (`gen_default_value_resolved` resolves the symbol via `ns` and emits
// a properly zeroed struct constant).
contract C {
    struct S { uint a; uint b; }
    S[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push();
        assert(arr.length == 1);
        assert(arr[0].a == 0);
        assert(arr[0].b == 0);
    }
}
