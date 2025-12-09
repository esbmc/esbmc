pragma solidity ^0.8.0;

struct T {
    bytes x;
    uint[3] y;
}
contract MyContract {
    uint8 i;
    T structJ;
    function getI() public returns (uint8) {
        return i;
    }
}
contract tupleABITrig {
    function test() public {
        MyContract tupleABIContract = new MyContract();
        MyContract tupleABIContract2 = new MyContract();
        assert(tupleABIContract.getI() != tupleABIContract2.getI());
    }
}
