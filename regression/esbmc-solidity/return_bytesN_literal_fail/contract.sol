// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Dual of return_bytesN_literal_pass: same lowered `return 0;` path, but
// the assertion expects a different value, so verification must FAIL
// (not crash).
contract C {
    function f() public pure returns (bytes32) { return 0; }
    function check() public pure {
        bytes32 x = f();
        assert(x == bytes32(uint256(1)));
    }
}
