// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
import "./contract3.sol"; 

contract M is T {
    function test() external pure {
        assert(1 == 0);
    }
}