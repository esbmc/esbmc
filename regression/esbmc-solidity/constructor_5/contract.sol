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

    function empty(uint8 x, uint8 y) public {
        data = data + x;
        empty1();
        data = data + y;
    }
}

contract DD {
    function empty2() public {}

    function D() public {
        BB x = new BB(2, 0);
        x.empty(0, 2);
        x.empty(0, 2);
    }
}
