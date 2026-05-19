// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Nondet hash abstraction: hash call completes and returns a value.
// The assertion h == h is trivially true for any nondet value.
contract HashBasic {
    function test() public pure {
        uint256 h = uint256(keccak256(abi.encodePacked(uint256(42))));
        assert(h == h);
    }
}
