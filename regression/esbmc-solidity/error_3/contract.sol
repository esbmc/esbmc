pragma solidity >=0.6.0;

error tt(uint x, uint8);

contract Base {
    function test(uint x, uint8 y) public pure {}

    function test1() public pure {
        test(1, 2);
        revert("test");
        assert(0 == 1);
    }

    function test2() public pure {
        test(1, 2);
        require(false);
        assert(0 == 1);
    }
}
