pragma solidity ^0.8.24;

contract Demo {
    uint public x = 1;

    function f() external pure {
        uint y = 10;

        {
            assert(y == 10);
            uint y = 20;
            assert(y == 20);
        }

        unchecked {
            assert(y == 10);
            uint y = 30;
            assert(y == 30);
        }
    }
}
