// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.decode with unbound (nondet) input bytes.
// Decoded value is nondet — asserting a specific value must FAIL.
contract AbiDecodeUnbound {
    function test(bytes memory data) public pure {
        uint256 decoded = abi.decode(data, (uint256));
        assert(decoded == 0); // must FAIL: decoded is nondet, can be any value
    }
}
