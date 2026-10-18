// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// CORE: `int[]` (signed-int) element coverage. Element-type-coverage
// gap before this test — the regression suite previously had no signed
// integer dyn-array push case.
contract C {
    int[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push();
        assert(arr.length == 1);
        assert(arr[0] == 0);
    }
}
