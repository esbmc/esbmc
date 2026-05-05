// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.decode: encode-decode with guard.
// Decoded value is nondet, but we can use require to constrain it.
contract AbiDecodeGuard {
    function test() public pure {
        bytes memory data = abi.encode(uint256(10));
        uint256 decoded = abi.decode(data, (uint256));
        require(decoded > 0);
        assert(decoded > 0); // must hold: guarded by require
    }
}
