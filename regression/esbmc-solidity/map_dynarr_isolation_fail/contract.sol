// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Isolation fail dual: asserting m[b].length != 0 after pushing only
// to m[a] must be refuted.
contract C {
    mapping(address => uint256[]) m;
    function test() public {
        m[address(0x1)].push(42);
        assert(m[address(0x2)].length != 0);
    }
}
