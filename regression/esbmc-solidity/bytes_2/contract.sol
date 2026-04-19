pragma solidity >=0.5.0;

contract Base {
    bytes3 data3;

    constructor() {
        data3 = 0;
    }

    function test() public {
        assert(data3 != 0x00); // should be 0
    }
}

contract Dreive {
    Base x = new Base();

    function test1() public {
        x.test();
    }
}
