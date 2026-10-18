// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Violation dual of multi_dim_fixed_addr_2d_pass. Writes one cell;
// asserts a false value at a different cell.
contract MultiDimAddr2DFail {
    address[3][2] internal grid;

    function run() external {
        grid[0][0] = address(0x1111);
        // BUG: grid[1][2] was never written.
        assert(grid[1][2] == address(0xbeef));
    }
}
