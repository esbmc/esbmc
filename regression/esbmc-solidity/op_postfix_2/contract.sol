// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Adversarial test: verification must FAIL.
// The contract asserts values that would only hold under prefix semantics.
// Under correct postfix semantics, the assertions are wrong.
contract PostfixFail {
    function test_postfix_fail() public pure {
        // If x++ were prefix (returning x+1), captured would be 6.
        // Under correct postfix semantics, captured is 5.
        uint x = 5;
        uint captured = x++;
        assert(captured == 6);  // WRONG: should be 5, must fail

        // Also test decrement: if y-- were prefix, snapped would be 9.
        // Under correct postfix, snapped is 10.
        uint y = 10;
        uint snapped = y--;
        assert(snapped == 9);   // WRONG: should be 10, must fail
    }
}
