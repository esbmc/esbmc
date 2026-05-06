// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodePacked with unbound (nondet) input.
// Encoding is deterministic over the same symbolic variable.
contract AbiEncodePackedUnbound {
    function test(uint256 x) public pure {
        uint256 h1 = uint256(keccak256(abi.encodePacked(x)));
        uint256 h2 = uint256(keccak256(abi.encodePacked(x)));
        assert(h1 == h2); // deterministic
    }
}
