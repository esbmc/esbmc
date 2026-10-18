// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// 2D fully-fixed `bool[N][M]` round-trip. Exercises option B native
// path with a 1-bit scalar element.
contract MultiDimBool2DPass {
    bool[3][2] internal flags;

    function pin() internal {
        flags[0][0] = true;
        flags[0][1] = false;
        flags[0][2] = true;
        flags[1][0] = false;
        flags[1][1] = true;
        flags[1][2] = false;
    }

    function run() external {
        pin();
        assert(flags[0][0] == true);
        assert(flags[0][1] == false);
        assert(flags[1][1] == true);
        flags[0][1] = true;
        assert(flags[0][1] == true);
        assert(flags[0][2] == true);  // untouched
    }
}
