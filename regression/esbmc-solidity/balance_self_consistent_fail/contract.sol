// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative counterpart to balance_self_consistent_pass: asserts that
// two consecutive reads of `address(this).balance` are NOT equal.  After
// the fix, both reads alias `this->$balance` (the same SSA cell), so the
// assertion is unsat — the harness reports a violation.
contract Bal {
    constructor() payable {}

    function probe() public view {
        uint a = address(this).balance;
        uint b = address(this).balance;
        assert(a != b);
    }
}
