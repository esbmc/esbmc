// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    uint x;
    uint y;

    constructor() {
        (x, y) = (1, 2);
    }

    function swap() external {
        (x, y) = (y, x);
    }

    function return_swap() internal view returns (uint, uint) {
        return (x, y);
    }

    function r_1() internal returns (uint) {
        return 1;
    }

    function test() public {
        uint xx;
        uint yy;
        (xx, yy) = return_swap();
        (xx, yy);
        (xx, yy) = (r_1(), r_1());
    }
    function test2() public returns (uint, uint) {
        if (true) return (1, 2);
        else return (3, 4);
    }
}

contract D is Base
{
    constructor()
    {
        (x,y) = (y,x);
    }
}