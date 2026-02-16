// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base{
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
}

contract Derived{
    function test_op() public {
        Base y = new Base();
        int x = 10;
        uint16 xx = 10;

        y.BO_Compound(x);
    }
}