// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test: string.concat with 3 args, nondet input makes assert fail
contract Test {
    function check(bool b) public pure {
        string memory s = string.concat("x", "y", "z");
        // s exists; use nondet bool to force failure
        if (b) {
            assert(false);
        }
    }
}
