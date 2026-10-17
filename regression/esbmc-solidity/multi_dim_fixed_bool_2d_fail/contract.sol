// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract MultiDimBool2DFail {
    bool[3][2] internal flags;

    function run() external {
        flags[0][0] = true;
        // BUG: flags[1][2] never written, must still be false.
        assert(flags[1][2] == true);
    }
}
