// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;
contract D {
    uint public x;
    constructor(uint a) payable {
        payable(msg.sender).transfer(1 ether);
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
        uint balancebefore = address(this).balance;
        D newD = new D{value: amount}(arg);
        uint balanceafter = address(this).balance;
        assert(balanceafter == balancebefore - amount);
    }
}

// esbmc/esbmc#4715: `newD.x()` returns a value from a void-typed call site, so
// convert_return takes its "function should not return value" diagnostic arm.
// The native return arm has to delegate that shape rather than drop the
// diagnostic. Comments go at the end: the committed .solast pins line numbers.
