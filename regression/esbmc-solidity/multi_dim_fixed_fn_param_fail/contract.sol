// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract MultiDimFnParam2DFail {
    function compute(uint256[3][2] memory grid) external pure returns (uint256) {
        grid[0][0] = 100;
        // BUG: grid[1][1] was not set to 999.
        assert(grid[1][1] == 999);
        return grid[1][1];
    }
}
