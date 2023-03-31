// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract BB {
    uint8 data;

    constructor(uint8 x, uint8 y) {
        data = x + y;
    }

    function empty1() public {
        assert(data == 2);
    }

    function empty(uint8 x, uint256 y) public {
        data = data + x;
        empty1();
    }
}

contract DD {
    BB x = new BB(2, 0);

    function empty2() public {}

    function D() public {
        x.empty(0, 2);
    }
}
