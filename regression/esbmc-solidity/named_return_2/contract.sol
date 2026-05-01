// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Test: named return parameter — default zero value (VERIFICATION FAILED)
contract NamedReturnFail {
    function getDefault() public pure returns (uint result) {
        // result is zero-initialized, never assigned
    }

    function test() public pure {
        uint v = getDefault();
        // v == 0 by default, but asserting v == 1 should fail
        assert(v == 1);
    }
}
