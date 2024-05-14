pragma solidity ^0.8.0;

contract MyContract {
    struct T {
        bytes4 bytesX;
        bytes4 bytesY;
    } 
    bytes4 testByte = "test";
    function callVuln() public {
        T memory tInstance = T(testByte,0x74657374);
        assert(tInstance.bytesX != tInstance.bytesY);
    }
}
