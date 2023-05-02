// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.0;

contract SolidityTest {
    int x;
    int y;

    constructor() {
        x = 1;
        y = 2;
    }
    
    function test_op() public{
        int result = x>y ? x+10: y+10;
        assert(x==11);
    }
}


