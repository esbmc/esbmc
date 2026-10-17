// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// 2D fully-fixed local variable inside a function body. Exercises
// option B path for stack/local allocation (as opposed to contract
// struct field or state-var).
contract MultiDimLocal2DPass {
    function run() external pure {
        uint256[3][2] memory grid;
        grid[0][0] = 10;
        grid[0][1] = 20;
        grid[0][2] = 30;
        grid[1][0] = 40;
        grid[1][1] = 50;
        grid[1][2] = 60;

        assert(grid[0][0] == 10);
        assert(grid[1][2] == 60);
        assert(grid[0][2] == 30);
        grid[1][0] = 999;
        assert(grid[1][0] == 999);
        assert(grid[0][0] == 10);  // untouched
    }
}
