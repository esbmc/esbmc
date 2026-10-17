// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// 2D fully-fixed `int256[N][M]` round-trip with negative values.
// Exercises option B native path with signed scalar element.
contract MultiDimInt2DPass {
    int256[3][2] internal grid;

    function pin() internal {
        grid[0][0] = -10;
        grid[0][1] = 20;
        grid[0][2] = -30;
        grid[1][0] = 40;
        grid[1][1] = -50;
        grid[1][2] = 60;
    }

    function run() external {
        pin();
        assert(grid[0][0] == -10);
        assert(grid[0][2] == -30);
        assert(grid[1][1] == -50);
        grid[1][1] = 77;
        assert(grid[1][1] == 77);
        assert(grid[0][0] == -10);  // untouched
    }
}
