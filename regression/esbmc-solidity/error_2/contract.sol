// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

error errmsg(int num1, uint num2, uint[2] addrs);

// Creating a contract
contract rStatement {
    error errmsg2();

    // Defining a function to check condition
    function checkOverflow(uint num1, uint num2) public view {
        uint sum = num1 + num2;
        uint[2] memory xx = [uint(1), 2];
        if (sum < 0 || sum > 255) {
            revert errmsg(int(num1), num2, xx);
        }
    }

    function checkequal(uint num1, uint num2) public view {
        require(num1 != num2, "not equal");
        if (num1 == num2) revert("equal");
        assert(num1 == num2);
    }
}
