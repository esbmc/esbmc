// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.0;

contract Base {
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

    function BO_Compound(int x) public {
        x += 10;
        assert(x == 20); // BO_AddAssign
        x -= 10;
        assert(x == 10); // BO_SubAssign
        x *= 10;
        assert(x == 100); // BO_MulAssign
        x /= 10;
        assert(x == 10); // BO_DivAssign
        x %= 9;
        assert(x == 1); // BO_RemAssign
        x <<= 2;
        assert(x == 4); // BO_ShlAssign
        x >>= 1;
        assert(x == 2); // BO_ShrAssign
        x &= 0;
        assert(x == 0); // BO_AndAssign
        x ^= 1;
        assert(x == 1); // BO_XorAssign
        x |= 1;
        assert(x == 1); // BO_OrAssign
    }

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

        y.BO_Cal(x);
        y.BO_Cmp(x);
        y.BO_Compound(x);
        y.UO_PreDec(--x);
        y.UO_PreInc(++x);
        y.UO_PostDec(x--);
        y.UO_PostInc(x++);
        y.UO_Minus(-x);
        y.UO_Not(0);
        y.UO_LNot(x);
    }
}
