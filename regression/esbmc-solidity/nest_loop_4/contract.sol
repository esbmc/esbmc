// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;

contract Base {
    int8 x;

    function test() public {
        x = 20;
        while (x > 0) for (int j = 0; j < 10; j++) x--;
        assert(x == 0);
    }
}
