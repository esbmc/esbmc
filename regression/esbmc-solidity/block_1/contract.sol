// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;

contract Base {
    int8 x;
    int8 y;

    function test() public {
        for (int i = 0; i < 10; i++) x++;
        if (x == 10) x = 1;
        while (x-- >= 0) y /= x;
    }
}
