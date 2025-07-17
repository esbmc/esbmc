pragma solidity >=0.5.0;

contract Base {
    constructor() {}

    function test() public {
        bytes3 data3 = 0;
        assert(data3 == hex"00");
        assert(data3 == 0x00);
    }
}
contract Dreive {
    Base x = new Base();

    function test1() public {
        x.test();
    }
}