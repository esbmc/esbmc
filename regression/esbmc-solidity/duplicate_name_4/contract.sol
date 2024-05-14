// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

struct Base {
    int Base;
}

contract Base1 {
    struct Base {
        int Base;
        int base;
    }
    Base base = Base(1,3);
    function test() public view {
        uint Base = 4;
        {
            uint Base = 2;
            assert(Base == 2);
        }
        assert(base.Base == 1);
        assert(Base == 3);
    }
}
