// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// 2D fully-fixed array as a field inside a struct state variable.
// Exercises option B native path when the nested array is nested one
// level deeper inside a user-defined struct.
contract MultiDimStructField2DPass {
    struct Box {
        uint256[3][2] cells;
        uint256 tag;
    }

    Box internal b;

    function pin() internal {
        b.tag = 7;
        b.cells[0][0] = 10;
        b.cells[0][1] = 20;
        b.cells[0][2] = 30;
        b.cells[1][0] = 40;
        b.cells[1][1] = 50;
        b.cells[1][2] = 60;
    }

    function run() external {
        pin();
        assert(b.tag == 7);
        assert(b.cells[0][0] == 10);
        assert(b.cells[1][2] == 60);
        assert(b.cells[0][2] == 30);
        b.cells[1][0] = 999;
        assert(b.cells[1][0] == 999);
        assert(b.cells[0][0] == 10);  // untouched
        assert(b.tag == 7);  // sibling field untouched
    }
}
