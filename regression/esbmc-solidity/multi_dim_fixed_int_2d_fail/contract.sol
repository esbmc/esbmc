// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract MultiDimInt2DFail {
    int256[3][2] internal grid;

    function run() external {
        grid[0][0] = -10;
        // BUG: grid[1][2] never written.
        assert(grid[1][2] == -999);
    }
}
