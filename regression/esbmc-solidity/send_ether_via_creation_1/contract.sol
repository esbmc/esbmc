// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;
contract D {
    uint public x;
    constructor(uint a) payable {
        x = a;
    }
}

contract C {
    D d = new D(4); // will be executed as part of C's constructor

    function createD(uint arg) public {
        D newD = new D(arg);
        newD.x();
    }

    function createAndEndowD(uint arg, uint amount) public payable {
        // Send ether along with the creation
        uint balanceBefore = address(msg.sender).balance;
        D newD = new D{value: amount}(arg);
        uint balanceAfter = address(msg.sender).balance;
        assert (balanceAfter == balanceBefore);
    }
}



