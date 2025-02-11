// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Test {
    event Deposit(address indexed _from, bytes32 indexed _id, int8 _value); // delete
    function deposit(bytes32 _id) public payable {
        int8 x = 127;
        int8 y = 0;
        emit Deposit(msg.sender, _id, x / y); // skip()
    }
}
