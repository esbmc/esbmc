// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract call {
    function test() public pure returns (bytes memory) {
        bytes memory x = new bytes(10);
        return x;
    }

    function test_return() public pure {
        bytes memory tmp = test();
        assert(tmp[0] == 0x00);
        assert(test()[0] == 0x00);
    }
}
