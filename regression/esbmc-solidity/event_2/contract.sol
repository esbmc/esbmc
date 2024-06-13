// SPDX-License-Identifier: GPL-3.0 
pragma solidity >=0.5.0;

contract Test {
    event Deposit(address indexed _from, bytes32 indexed _id, uint8 _value);
    function deposit(bytes32 _id) public payable {
        uint8 x = 255;
        uint8 y = 2;
        emit Deposit(msg.sender, _id, x+y);
    }
}
