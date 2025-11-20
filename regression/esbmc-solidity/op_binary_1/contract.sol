// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base{
        function BO_Cal(int x) public {
        assert(x + 10 == 20); // BO_Add
        assert(x - 10 == 0); // BO_Sub
        assert(x * 10 == 100); // BO_Mul
        assert(x / 10 == 1); // BO_Div
        assert(x % 10 == 0); // BO_Rem
        assert(x << 1 == 20); // BO_Shl
        assert(x >> 1 == 5); // BO_Shr
        assert((x & 0) == 0); // BO_And
        assert(x ^ 0 == x); // BO_Xor
        assert(x | 0 == x); // BO_Or
    }

}

contract Derived{
    function test_op() public {
        Base y = new Base();
        int x = 10;
        uint16 xx = 10;

        y.BO_Cal(x);
    }
}