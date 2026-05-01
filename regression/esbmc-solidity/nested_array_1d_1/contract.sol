// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract DynArray1DPass {
    uint256[] internal arr;

    function run() external {
        assert(arr.length == 0);

        // push with value tracking
        arr.push(10);
        arr.push(20);
        arr.push(30);
        assert(arr.length == 3);
        assert(arr[0] == 10);
        assert(arr[1] == 20);
        assert(arr[2] == 30);

        // pop decreases length
        arr.pop();
        assert(arr.length == 2);
        assert(arr[0] == 10);
        assert(arr[1] == 20);

        // push after pop writes to correct index
        arr.push(999);
        assert(arr.length == 3);
        assert(arr[2] == 999);

        // pop all
        arr.pop();
        arr.pop();
        arr.pop();
        assert(arr.length == 0);

        // push again after full drain
        arr.push(42);
        assert(arr.length == 1);
        assert(arr[0] == 42);
    }
}
