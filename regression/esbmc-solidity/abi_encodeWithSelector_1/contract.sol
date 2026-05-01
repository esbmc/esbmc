// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeWithSelector with bound input.
// Same selector and data must produce same encoding hash.
contract AbiEncodeWithSelectorBound {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encodeWithSelector(bytes4(0x12345678), uint256(100))));
        uint256 h2 = uint256(keccak256(abi.encodeWithSelector(bytes4(0x12345678), uint256(100))));
        assert(h1 == h2); // deterministic
    }
}
