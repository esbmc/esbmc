// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Tests functional consistency of keccak256:
//   same input => same output (deterministic abstraction).
// This enables string equality comparison via hash.
contract HashConsistency {
    function test_same_input() public pure {
        uint256 a = 42;
        uint256 h1 = uint256(keccak256(abi.encodePacked(a)));
        uint256 h2 = uint256(keccak256(abi.encodePacked(a)));
        assert(h1 == h2); // must hold: same input => same output
    }

    function test_different_input() public pure {
        uint256 h1 = uint256(keccak256(abi.encodePacked(uint256(1))));
        uint256 h2 = uint256(keccak256(abi.encodePacked(uint256(2))));
        assert(h1 != h2); // must hold: bijective (no collisions)
    }
}
