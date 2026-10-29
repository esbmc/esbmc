// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bug A regression-lock: `delete dynArr` must reset the length companion
// `<arr>_dynarray_len[this->$address]`, not just the data array.
//
// Currently fails on the post-delete assertion. KNOWNBUG until the
// emit_delete_block helper threads the length-reset alongside the data
// reset.
contract C {
    uint[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push(1);
        arr.push(2);
        arr.push(3);
        assert(arr.length == 3);
        delete arr;
        assert(arr.length == 0);
    }
}
