// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test: string.concat with 3 arguments (variadic)
contract Test {
    function check() public pure {
        string memory s = string.concat("hello", " ", "world");
        bytes32 h = keccak256(abi.encodePacked(s));
        assert(h == h);
    }
}
