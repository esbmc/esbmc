// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract PrevRandao {
    function test() public view {
        uint256 r = block.prevrandao;
        // prevrandao is nondet — this specific equality cannot be guaranteed
        assert(r == 42);
    }
}
