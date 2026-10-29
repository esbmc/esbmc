// SPDX-License-Identifier: MIT
// Regression for Block 3: `--no-standard-checks` must imply
// `--no-narrowing-check` (i.e. the bulk-disable pattern includes
// narrowing). Same contract as narrowing_user_cast_fail; under
// `--no-standard-checks` verification should succeed.
pragma solidity >=0.8.0;

contract T {
    function narrow(uint256 x) public pure returns (uint8) {
        return uint8(x);
    }
}
