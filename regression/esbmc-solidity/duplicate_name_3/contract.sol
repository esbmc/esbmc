// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

struct Base {
    int Base;
}

contract Base1 {
    struct Base {
        int Base;
    }
    Base base = Base(1);
    function test() public view {
        int Base = 2;
        assert(base.Base == 1);
        assert(Base == 2);
    }
}
