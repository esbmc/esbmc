// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Robot {
    int x = 0;
    int y = 0;

    constructor() {}

    function moveLeftUp() public {
        --x;
        ++y;
    }

    function moveLeftDown() public {
        --x;
        --y;
    }

    function moveRightUp() public {
        ++x;
        ++y;
    }

    function moveRightDown() public {
        ++x;
        --y;
    }

    function moveRight2Down() public {
        ++x;
        --y;
    }

    function test2() public {
        require(x == 0);
        require(y == 0);
        assert(x == 0 && y == 1);
    }

    function test() public {
        assert((x + y) % 2 == 0);
    }
}
