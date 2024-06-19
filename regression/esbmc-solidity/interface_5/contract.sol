pragma solidity >=0.5.0;

abstract contract caculator {
    enum E {
        x,
        y
    }
}

contract Test {
    function caculator() external pure{
        assert(caculator.E.x <= caculator.E.y);
    }
    function test() external pure{
        assert(caculator.E.x > caculator.E.y);
    }
}