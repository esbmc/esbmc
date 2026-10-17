// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Unbound fail dual.
contract C {
    mapping(address => uint256[]) m;
    function test() public {
        m[address(0x1)].push(42);
        assert(m[address(0x1)][0] != 42);
    }
}
