// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract ArrayBoundary {
    function test_static_array() public pure {
        uint256[5] memory arr;
        arr[0] = 10;
        arr[1] = 20;
        arr[2] = 30;
        arr[3] = 40;
        arr[4] = 50;

        assert(arr[0] == 10);
        assert(arr[4] == 50);

        // Sum
        uint256 sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4];
        assert(sum == 150);
    }

    function test_array_overwrite() public pure {
        uint256[3] memory a;
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[0] = 99;
        assert(a[0] == 99);
        assert(a[1] == 2);
    }
}
