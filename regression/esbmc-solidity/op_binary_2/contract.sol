// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.0;

contract Base{


    function BO_Cmp(int x) public {
        assert(x > 9); // BO_GT
        assert(x < 11); // BO_LT
        assert(x >= 10); // BO_GE
        assert(x <= 10); // BO_LE
        assert(x != 9); // BO_NE
        assert(x == 10); // BO_EQ
        assert(!(true && false)); // BO_LAnd
        assert(true || false); // BO_LOr
    }
}

contract Derived{
    function test_op() public {
        Base y = new Base();
        int x = 10;
        uint16 xx = 10;

        y.BO_Cmp(x);
    }
}