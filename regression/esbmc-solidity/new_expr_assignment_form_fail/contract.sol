// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

// FAIL dual of new_expr_assignment_form_pass: the getter on an assignment-form
// `new C()` instance now executes precisely (x == 7), so a claim that it equals
// 8 is a genuine violation.
contract C {
    uint256 public x = 7;
}

contract Harness {
    C a;
    constructor() {
        a = new C();
    }
    function check() public view {
        assert(a.x() == 8);   // x is precisely 7 -> violated
    }
}
