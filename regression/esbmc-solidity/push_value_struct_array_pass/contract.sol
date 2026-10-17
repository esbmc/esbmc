// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// CORE: `arr.push(S(7, 13))` — push-with-explicit-value works for
// struct elements today (the rhs is constructed by the caller; no
// gen_zero involved). Locks this so S2's `gen_default_value_resolved`
// only fires on the no-arg branch.
contract C {
    struct S { uint a; uint b; }
    S[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push(S(7, 13));
        assert(arr.length == 1);
        assert(arr[0].a == 7);
        assert(arr[0].b == 13);
    }
}
