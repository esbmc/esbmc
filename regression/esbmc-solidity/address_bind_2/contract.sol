// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

interface T {
    function test() external;
}

contract M is T {
    function test() external pure {
        assert(1 == 0);
    }
}

contract Base {
    T y;
    constructor(address t) {
        y = T(t);
    }
    function test() public {
        y.test();
    }
}
