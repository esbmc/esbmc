// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract MultiDimStructField2DFail {
    struct Box {
        uint256[3][2] cells;
        uint256 tag;
    }

    Box internal b;

    function run() external {
        b.cells[0][0] = 10;
        // BUG: b.cells[1][2] never written.
        assert(b.cells[1][2] == 42);
    }
}
