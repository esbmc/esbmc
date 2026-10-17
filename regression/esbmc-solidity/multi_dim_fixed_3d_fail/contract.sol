// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Violation dual of multi_dim_fixed_3d_pass. Writes one cell, asserts a
// stale non-zero value at an unrelated cell. If cross-row isolation is
// sound, ESBMC should FAIL on the stale-value claim; if it regressed
// back to the old `T***` aliasing pattern the false claim would succeed.
contract MultiDimFixed3DFail {
    uint256[2][3][4] internal g;

    function run() external {
        g[0][0][0] = 100;
        // BUG: g[3][2][1] was never written; it must still be 0.
        assert(g[3][2][1] == 42);
    }
}
