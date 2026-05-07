// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract DynArray2DFail {
    uint256[][] internal a2;

    function run() external {
        a2.push();
        a2.push();
        a2.push();

        // outer length is 3
        a2.pop();
        // BUG: length is now 2, not 3
        assert(a2.length == 3);
    }
}
