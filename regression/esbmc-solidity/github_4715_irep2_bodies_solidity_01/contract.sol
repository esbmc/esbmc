// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.decode with bound input: encode then decode round-trip.
// With nondet decode model, decoded value is unconstrained.
contract AbiDecodeBound {
    function test() public pure {
        bytes memory data = abi.encode(uint256(42));
        uint256 decoded = abi.decode(data, (uint256));
        assert(decoded <= type(uint256).max); // trivially true
    }
}
