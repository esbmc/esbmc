// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Length fail dual: three pushes must NOT yield length != 3.
contract C {
    mapping(address => uint256[]) m;
    function test() public {
        m[address(0x1)].push(1);
        m[address(0x1)].push(2);
        m[address(0x1)].push(3);
        assert(m[address(0x1)].length != 3);
    }
}
