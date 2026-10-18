// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to pass: same nested-mapping bytes32 key lowering, plus an
// unconditional assert(false) placed ahead of the mapping write.
// The bytes32 mapping lowering still runs during GOTO generation (so
// if the fix were absent the frontend would abort before symex), but
// the assert comes first so slicing does not elide the VCC.

contract C {
    mapping(address => mapping(bytes32 => uint256)) private store;

    function go() external {
        assert(false);
        address a = address(0x1);
        bytes32 k = bytes32(uint256(1));
        store[a][k] = 42;
    }
}
