// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encode with bound (concrete) input.
// abi.encode(42) should be deterministic: encoding same value twice yields same result.
contract AbiEncodeBound {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encode(uint256(42))));
        uint256 h2 = uint256(keccak256(abi.encode(uint256(42))));
        assert(h1 == h2); // same input => same encoding => same hash
    }
}
