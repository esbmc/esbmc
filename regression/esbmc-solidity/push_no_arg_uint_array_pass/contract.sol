// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// CORE: `uint[] arr; arr.push();` works today — primitive `uint256`
// element type goes through gen_zero(unsignedbv) which is correctly
// handled in expr_util.cpp.  Locks the baseline so the upcoming
// `gen_default_value_resolved` refactor (S1+S2) doesn't regress it.
contract C {
    uint[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push();
        assert(arr.length == 1);
        assert(arr[0] == 0);
    }
}
