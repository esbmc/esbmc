// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodePacked with bound (concrete) input.
// Packed encoding of the same value must be deterministic.
contract AbiEncodePackedBound {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encodePacked(uint256(100))));
        uint256 h2 = uint256(keccak256(abi.encodePacked(uint256(100))));
        assert(h1 == h2); // same input => same output
    }
}
