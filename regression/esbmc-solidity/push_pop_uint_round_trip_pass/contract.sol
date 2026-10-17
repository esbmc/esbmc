// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// CORE: push 3 elements, pop 1, verify length and remaining elements.
// Locks the basic non-empty pop behaviour against any side-effects of
// S3 (length=0 assume) and S4 (implicit-delete).
contract C {
    uint[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push(10);
        arr.push(20);
        arr.push(30);
        assert(arr.length == 3);
        arr.pop();
        assert(arr.length == 2);
        assert(arr[0] == 10);
        assert(arr[1] == 20);
    }
}
