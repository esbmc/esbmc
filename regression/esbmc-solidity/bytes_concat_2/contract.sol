// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test: bytes.concat result length is nonzero when inputs are nonzero
contract Test {
    bytes public data;

    function check() public {
        bytes memory a = new bytes(2);
        bytes memory b = new bytes(3);
        bytes memory c = bytes.concat(a, b);
        // c.length should be 5, asserting 0 should fail
        assert(c.length == 0);
    }
}
