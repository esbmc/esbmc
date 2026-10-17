// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Foundry coverage-test generation under --k-induction. Collection happens in
// the per-phase base-case bmct while the coverage report is emitted by the
// k-induction strategy driver; without threading the strategy-level generator
// into the base-case bmct, the collected cases are discarded and no test is
// produced. This locks the fix: k-induction must still generate the test.
contract C {
    bool public hit;
    function poke(uint256 x) external {
        if (x == 42) hit = true;
    }
}
