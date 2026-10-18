// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B1 fail dual — verifies that an explicit assertion violation
// reachable through a successful (non-reverting) path is still
// detected after the revert state-rollback rework.  The require
// admits values up to 999, so the assert(x < 50) is reachable with
// x in [50, 999] — must FAIL.
contract C {
    uint public x;

    function setIfBounded(uint v) public {
        require(v < 1000, "out of range");
        x = v;
    }

    function check() public view {
        assert(x < 50);
    }
}
