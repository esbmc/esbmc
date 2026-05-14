// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Nondet hash abstraction: hash output is NOT the identity of the input.
// With nondet model the solver finds h != 42, so the assertion fails.
contract HashNotIdentity {
    function test() public pure {
        uint256 x = 42;
        uint256 h = uint256(keccak256(abi.encodePacked(x)));
        assert(h == x);
    }
}
