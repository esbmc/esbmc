// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Companion FAIL case: same source as dynamic_balance_payback_pass but
// asserts the naïve `post == pre - amount` (no payback correction).
// Should be falsified at 1-ether magnitude — locking in the bug-detection
// that was silently lost when `nondet_uint()` was 32-bit.

contract D {
    constructor() payable {
        payable(msg.sender).transfer(1 ether);
    }
}

contract C {
    function check(uint256 amount) external payable {
        uint256 pre = address(this).balance;
        D d = new D{value: amount}();
        uint256 post = address(this).balance;
        assert(post == pre - amount);   // wrong — D pays back 1 ether
    }
}
