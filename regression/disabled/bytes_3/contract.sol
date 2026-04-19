pragma solidity >=0.5.0;

contract Base {
    bytes16 data1;
    bytes4 data2;

    constructor() {
        data1 = "test";
        data2 = 0x74657374; // "test"
    }

    function test() public {
        data1<<=8;
        assert(data1 == "est");
        assert((data1<<8) == "st");
        
        assert(data2>>8 == 0x00746573);
        
        // cut off
        assert(data2<<8 == 0x65737400);
    }
}

contract Dreive {
    Base x = new Base();

    function test1() public {
        x.test();
    }
}
