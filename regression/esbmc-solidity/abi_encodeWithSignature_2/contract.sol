// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeWithSignature with unbound (nondet) input.
// Same signature + same nondet arg => same encoding hash.
contract AbiEncodeWithSignatureUnbound {
    function test(uint256 x) public pure {
        uint256 h1 = uint256(keccak256(abi.encodeWithSignature("foo(uint256)", x)));
        uint256 h2 = uint256(keccak256(abi.encodeWithSignature("foo(uint256)", x)));
        assert(h1 == h2); // deterministic
    }
}
