// SPDX-License-Identifier: MIT
// Regression for Block 2: `--no-narrowing-check` must be a working
// dedicated toggle that silences narrowing checks on user-written
// casts. Same contract as narrowing_user_cast_fail; with the flag
// passed, verification should succeed instead of failing.
pragma solidity >=0.8.0;

contract T {
    function narrow(uint256 x) public pure returns (uint8) {
        return uint8(x);
    }
}
