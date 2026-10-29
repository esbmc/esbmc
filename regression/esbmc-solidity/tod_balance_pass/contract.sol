// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Both functions transfer fixed amounts to EOA addresses.  Net balance
// change is the sum of the two amounts independent of order, so
// address(this).balance is equal in both orderings.
//
// Pins the negative side of TOD-Balance: candidate analysis must place
// __balance in W for both functions, the harness must emit the balance
// equality assertion, and the assertion must hold (no order
// dependence).
contract Bal {
    constructor() payable {}

    function payA(address payable to) public {
        if (to == address(this)) return;
        to.transfer(10);
    }

    function payB(address payable to) public {
        if (to == address(this)) return;
        to.transfer(20);
    }
}
