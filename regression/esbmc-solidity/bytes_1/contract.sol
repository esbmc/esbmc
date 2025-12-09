pragma solidity >=0.5.0;

contract Base {
    bytes16 data1;
    bytes4 data2;

    constructor() {
        data1 = "test";
        data2 = 0x74657374; // "test"
    }

    function test() public {
        assert(data1 == data2);
        bytes2 data4 = bytes2(data2);
        assert(data4 == 0x7465);
        bytes4 data5 = bytes4(data4);
        assert(data5 == 0x74650000);
    }
}

contract Dreive {
    Base x = new Base();

    function test1() public {
        x.test();
    }
}
