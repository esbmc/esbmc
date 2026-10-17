// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Solidity spec: `delete a[i]` on a dynamic array resets only element i,
// keeps `length` unchanged (leaves a "gap" — spec wording).
//
// CORE: locks the element-keeps-length behaviour. Currently passes
// because `delete a[i]` lowers to `a[i] = 0` (element write); the
// length companion `<arr>_dynarray_len[$address]` is untouched. The
// upcoming refactor must not regress this.
contract C {
    uint[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push(10);
        arr.push(20);
        arr.push(30);
        delete arr[1];
        assert(arr.length == 3);
        assert(arr[0] == 10);
        assert(arr[1] == 0);
        assert(arr[2] == 30);
    }
}
