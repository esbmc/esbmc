// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// The generated per-contract harness calls nondet_bool() as its loop condition.
// Without the callee refresh and the return-type alignment, that call reaches
// goto_convert with no type at all and its branch guard is nil.
contract C {
    function f() public pure {
        uint a = 1;
        assert(a == 1);
    }
}
