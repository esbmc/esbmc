// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// CORE: `bool[]` element coverage. Bool defaults to false; locks the
// gen_zero(bool) → false path through the post-S1 helper.
contract C {
    bool[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push();
        assert(arr.length == 1);
        assert(arr[0] == false);
    }
}
