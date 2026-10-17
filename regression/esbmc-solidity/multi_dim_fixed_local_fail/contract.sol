// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract MultiDimLocal2DFail {
    function run() external pure {
        uint256[3][2] memory grid;
        grid[0][0] = 10;
        // BUG: grid[1][2] never written.
        assert(grid[1][2] == 42);
    }
}
