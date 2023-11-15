// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.0;

contract Base {
    function UO_Minus(int x) public {
        assert(x == -10);
    }

    function UO_Not(uint16 x) public {
        assert(~x == 65535);
    }

    function UO_LNot(int x) public {
        assert(!(x == 0));
    }
}

contract Derived {
    function test_op() public {
        Base y = new Base();
        int x = 10;
        uint16 xx = 10;

        y.UO_Minus(-x);
        y.UO_Not(0);
        y.UO_LNot(x);
    }
}
