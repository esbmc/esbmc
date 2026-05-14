// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract BlobHashTest {
    function testHash() public view returns (bytes32) {
        return blobhash(0);
    }

    function test() public view {
        // blobbasefee is nondet — cannot guarantee zero
        uint256 fee = block.blobbasefee;
        assert(fee == 0);
    }
}
