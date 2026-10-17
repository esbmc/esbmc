// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.4;

// Tests that revert only guards one path — the other path can still fail.
error TooLarge(uint256 value, uint256 max);

contract RevertFail {
    function check(uint256 x) public pure {
        if (x > 100)
            revert TooLarge(x, 100);
        // x <= 100 here, but the assertion below is wrong
        assert(x < 50);  // SHOULD FAIL: x could be 50..100
    }
}
