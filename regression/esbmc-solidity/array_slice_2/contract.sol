// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

contract SliceFail {
    function test(bytes calldata data) external pure {
        if (data.length >= 4) {
            bytes calldata first4 = data[:4];
            // slice is nondet: cannot guarantee length == 4
            assert(first4.length == 4);
        }
    }
}
