// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.0;

contract Base {
    function UO_PreDec(int x) public {
        assert(x == 9);
    }

    function UO_PreInc(int x) public {
        assert(x == 10);
    }

    function UO_PostDec(int x) public {
        assert(x == 10);
    }

    function UO_PostInc(int x) public {
        assert(x == 9);
    }

}

contract Derived {
    function test_op() public {
        Base y = new Base();
        int x = 10;
        uint16 xx = 10;

        y.UO_PreDec(--x);
        y.UO_PreInc(++x);
        y.UO_PostDec(x--);
        y.UO_PostInc(x++);
    }
}
