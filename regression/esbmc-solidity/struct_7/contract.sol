pragma solidity ^0.8.0;

contract MyContract {
    struct T {
        bytes4 bytesX;
        uint[4] x;
    }
    bytes4 testByte = "abcd";
    function callVuln() public {
        T memory tInstance = T("abcd", [uint(1), 2, 3, 4]);
        assert(tInstance.bytesX[0] == tInstance.bytesX[0]);
        assert(tInstance.bytesX[3] == tInstance.bytesX[3]);
    }
}
