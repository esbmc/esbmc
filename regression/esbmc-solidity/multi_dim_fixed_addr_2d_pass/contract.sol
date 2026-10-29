// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// 2D fully-fixed `address[N][M]` round-trip. Exercises option B's
// native nested array_typet path with a non-uint scalar element type.
contract MultiDimAddr2DPass {
    address[3][2] internal grid;

    function pin() internal {
        grid[0][0] = address(0x1111);
        grid[0][1] = address(0x2222);
        grid[0][2] = address(0x3333);
        grid[1][0] = address(0x4444);
        grid[1][1] = address(0x5555);
        grid[1][2] = address(0x6666);
    }

    function run() external {
        pin();
        assert(grid[0][0] == address(0x1111));
        assert(grid[0][2] == address(0x3333));
        assert(grid[1][1] == address(0x5555));
        grid[1][1] = address(0xdead);
        assert(grid[1][1] == address(0xdead));
        assert(grid[0][0] == address(0x1111));
    }
}
