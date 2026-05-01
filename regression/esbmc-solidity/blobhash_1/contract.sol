// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract BlobHashTest {
    function test() public view {
        bytes32 h = blobhash(0);
        // blobhash returns nondet bytes32, any value is valid
        assert(h == h);
    }
}
