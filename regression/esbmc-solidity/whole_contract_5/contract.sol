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

    function test(uint8 z) public {
        assert(z==0);
    }
}