// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Isolation between distinct keys: pushing to m[a] must not affect
// m[b].length, which should remain 0.
contract C {
    mapping(address => uint256[]) m;
    function test() public {
        m[address(0x1)].push(42);
        assert(m[address(0x2)].length == 0);
    }
}
