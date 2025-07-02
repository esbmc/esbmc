// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SelfDestructExample {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    receive() external payable {}

    function getBalance() external view returns (uint) {
        return address(this).balance;
    }

    function destroy(address payable _to) external {
        require(msg.sender == owner, "Only owner can destroy the contract");
        selfdestruct(_to);
    }
}
