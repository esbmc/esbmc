// SPDX-License-Identifier: MIT
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

    function inv() public view {
        assert((x == 2 && y == 4));
    }

    function test() public view {
        assert((x + y) % 2 == 0);
    }
}
