// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;

contract Base {
    uint[] public a;

    function SelectionSort(uint array_size) public {
        uint i;
        uint j;
        uint min;
        uint temp;
        for (i = 0; i < array_size - 1; ++i) {
            min = i;
            for (j = i + 1; j < array_size; ++j) {
                if (a[j] < a[min]) min = j;
            }
            temp = a[i];
            a[i] = a[min];
            a[min] = temp;
        }
    }

    function test() public {
        uint array_size = 100;
        SelectionSort(array_size);
    }
}