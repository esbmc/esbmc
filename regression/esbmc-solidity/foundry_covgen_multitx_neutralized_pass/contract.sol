// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Bank {
    uint256 public balance;
    uint256 public paid;
    function deposit() external { balance += 100; }
    function withdraw() external {
        if (balance >= 50) { balance -= 50; paid = 1; }
        else { paid = 2; }
    }
}
