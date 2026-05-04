// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test: bytes.concat with 2 arguments
contract Test {
    bytes public data;

    function check() public {
        bytes memory a = new bytes(2);
        bytes memory b = new bytes(3);
        bytes memory c = bytes.concat(a, b);
        assert(c.length == c.length);
    }
}
