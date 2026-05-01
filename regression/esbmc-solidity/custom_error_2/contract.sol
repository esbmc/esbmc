// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.27;

/// Insufficient balance for transfer.
error InsufficientBalance(uint256 available, uint256 required);

// Test custom error revert with --bound mode.
contract TestToken {
    mapping(address => uint) balance;
    function transferWithRevertError(address to, uint256 amount) public {
        if (amount > balance[msg.sender])
            revert InsufficientBalance({
                available: balance[msg.sender],
                required: amount
            });
        balance[msg.sender] -= amount;
        balance[to] += amount;
    }
    function transferWithRequireError(address to, uint256 amount) public {
        require(amount <= balance[msg.sender], InsufficientBalance(balance[msg.sender], amount));
        balance[msg.sender] -= amount;
        balance[to] += amount;
    }
}
