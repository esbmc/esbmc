// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;

contract Base {
    int8 x;

    function test() public
    {
        x=0;
        for(int i =0; i<10;i++)
            for(int j=0; j<10;j++)
                x++;    // overflow
        assert(x==100);
    }
}