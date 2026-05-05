// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract DynArray1DFail {
    uint256[] internal arr;

    function run() external {
        arr.push(10);
        arr.push(20);
        arr.push(30);

        // pop removes last, arr[1] should still be 20
        arr.pop();

        // BUG: arr[1] is 20 not 30 after pop
        assert(arr[1] == 30);
    }
}
