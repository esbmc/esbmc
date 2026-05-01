// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeWithSignature with bound input.
// Same signature + same data => same encoding hash.
contract AbiEncodeWithSignatureBound {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encodeWithSignature("transfer(uint256)", uint256(50))));
        uint256 h2 = uint256(keccak256(abi.encodeWithSignature("transfer(uint256)", uint256(50))));
        assert(h1 == h2); // deterministic
    }
}
