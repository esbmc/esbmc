// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pin-test (U5): 3D state-var dynamic array is not supported by
// the per-instance dyn-array model. The clone walker's
// _ESBMC_arrcpy is 1-level only (per reference_deep_copy_semantics);
// 3D write-then-read invariants do not hold. Documents the
// fallback. Reclassify CORE when 3D state-var dyn-arrays are
// supported (recursive walker + CVC5 tuple encoding).
contract C {
    uint256[][][] grid;

    function setup() public {
        grid = new uint256[][][](2);
        for (uint256 i = 0; i < 2; ++i) {
            grid[i] = new uint256[][](2);
            for (uint256 j = 0; j < 2; ++j)
                grid[i][j] = new uint256[](2);
        }
    }

    function test() public {
        setup();
        grid[1][1][1] = 7;
        // Under correct modelling read must return 7.
        assert(grid[1][1][1] == 7);
    }
}
