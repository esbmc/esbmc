// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encode with unbound (nondet) input.
// abi.encode is deterministic: encoding the same nondet variable twice must match.
contract AbiEncodeUnbound {
    function test(uint256 x) public pure {
        uint256 h1 = uint256(keccak256(abi.encode(x)));
        uint256 h2 = uint256(keccak256(abi.encode(x)));
        assert(h1 == h2); // deterministic: same x => same encoding
    }
}
