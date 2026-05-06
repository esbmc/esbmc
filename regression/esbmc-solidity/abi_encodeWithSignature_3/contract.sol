// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeWithSignature: different signatures must produce different hash.
// The identity model captures the first argument (signature string), so
// different signatures => different encoding.
contract AbiEncodeWithSignatureDiff {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encodeWithSignature("transfer(uint256)", uint256(1))));
        uint256 h2 = uint256(keccak256(abi.encodeWithSignature("approve(uint256)", uint256(1))));
        assert(h1 == h2); // must FAIL: different signatures
    }
}
