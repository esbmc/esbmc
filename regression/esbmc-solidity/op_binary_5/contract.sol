pragma solidity >=0.5.0;

contract Base {
}

contract Derive {
    Base x = new Base();

    function comp() public {
        uint8 a = 2;
        uint8 b = 3;
        uint8 c = a ** b;
        assert(c == 8);
    }
}