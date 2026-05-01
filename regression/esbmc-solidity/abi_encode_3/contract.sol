// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encode: different inputs must produce different encodings.
// This tests the injectivity of the identity abstraction.
contract AbiEncodeDifferent {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encode(uint256(1))));
        uint256 h2 = uint256(keccak256(abi.encode(uint256(2))));
        assert(h1 == h2); // must FAIL: different inputs
    }
}
