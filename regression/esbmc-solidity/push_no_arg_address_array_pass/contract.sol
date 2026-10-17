// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// CORE: `address[]` element coverage. `address` lowers to a 160-bit
// unsigned bitvector — primitive — so gen_zero works, and post-S2 the
// recursive helper handles it the same way.
contract C {
    address[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push();
        assert(arr.length == 1);
        assert(arr[0] == address(0));
    }
}
