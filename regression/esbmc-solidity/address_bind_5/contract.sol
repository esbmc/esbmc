// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

interface T {
    function test() external;
}

contract M is T {
    function test() external pure {
        assert(1 == 1);
    }
}

contract N {
    function test() external pure {
        assert(1 == 0);
    }
}

contract Base {
    T y;
    constructor(T z) {
        y = T(address(z)); // can be N, Solidity force type conversion
    }
    function test() public {
        y.test();
    }
}
