// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    struct tuple1 {
        uint x;
        uint8 y;
    }
    uint x;
    uint y;
    tuple1 xx;
    constructor() {
        (x, y) = (1, 2);
    }

    function name() public {
        assert(y == 2);
    }
}
