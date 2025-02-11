pragma solidity >=0.5.0;

contract Base {
    bytes16 data1;
    bytes4 data2;

    constructor() {
        data1 = "test";
        data2 = 0x74657374; // "test"
    }

    function test() public {
        assert(data1[1] == data2[1]);

        assert(data1[0] == 0x74);
        assert(data1[1] == 0x65);
        assert(data1[5] == 0x00);
        assert(data1[6] == 0x00);
        

        assert(data2[1] == 0x65);
    }
}

contract Dreive {
    Base x = new Base();

    function test1() public {
        x.test();
    }
}
