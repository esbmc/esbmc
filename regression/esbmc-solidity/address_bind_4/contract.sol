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
    address x = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;
    T y = T(x);
    T s;
    constructor(T z) {
        s = T(address(y));
    }
    function test() public {
        s.test();
    }
}
