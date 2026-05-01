// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeWithSelector with unbound (nondet) input.
// Same selector + same nondet arg => same encoding hash.
contract AbiEncodeWithSelectorUnbound {
    function test(uint256 x) public pure {
        uint256 h1 = uint256(keccak256(abi.encodeWithSelector(bytes4(0xaabbccdd), x)));
        uint256 h2 = uint256(keccak256(abi.encodeWithSelector(bytes4(0xaabbccdd), x)));
        assert(h1 == h2); // deterministic
    }
}
