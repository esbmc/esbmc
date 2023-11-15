// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract SolidityTest {

    function getResult() public view {
        uint a = 1; // local variable
        uint b = 2;
        uint result = (a > b ? a : b); //conditional operation
        assert(result == b);
        assert(result != a);
    }
}