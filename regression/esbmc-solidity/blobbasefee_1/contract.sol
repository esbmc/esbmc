// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract BlobTest {
    function test() public view {
        uint256 fee = block.blobbasefee;
        // blobbasefee is nondet, so >= 0 is always true for uint256
        assert(fee >= 0);
    }
}
