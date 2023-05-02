// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.0;

contract Base {
    function test_op() public {
        int x = 1;
        x /= --x; // division by zero
        assert(1==1);
    }
}
