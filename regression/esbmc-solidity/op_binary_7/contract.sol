pragma solidity >=0.5.0;

contract Base {
    function UO_Minus(int8 x) internal {
        assert(x == -10);
    }

    function UO_Not(uint16 x) internal {
        assert(~x == 65535);
    }

    function UO_LNot(int8 x) internal {
        assert(!(x == 0));
    }
}

contract Derived is Base {
    function test_op() public {
        int8 x = 10;
        uint16 xx = 10;

        UO_Minus(-x);
        UO_Not(0);
        UO_LNot(x);
    }
}
