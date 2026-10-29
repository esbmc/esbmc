// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Each re-entry through `to.call("")` raises `depth`, and the assertion
// fails once two transactions have re-entered deep enough. Cutting the
// re-entry off at the unwinding bound in the inductive step would hide the
// deeper calls and prove the assertion at k = 1; the step must instead
// havoc what they may write, leaving the base case to find the violation.
contract Nest {
    uint256 depth;

    function enter(address to) public {
        depth = depth + 1;
        to.call("");
        assert(depth < 5);
    }
}
