// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Tests that keccak256 functional consistency detects inequality.
// keccak256(1) == keccak256(2) should FAIL (different inputs, bijective hash).
contract HashFail {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encodePacked(uint256(1))));
        uint256 h2 = uint256(keccak256(abi.encodePacked(uint256(2))));
        assert(h1 == h2); // must FAIL: different inputs => different outputs
    }
}
