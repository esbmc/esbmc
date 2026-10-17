// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual of library_caller_balance_debit_pass.  The library debit
// fires deterministically, so an assertion claiming the delta is
// anything OTHER than 10 is refutable.
library Pay {
    function send(address payable to, uint256 val) internal {
        to.transfer(val);
    }
}

contract A {
    uint256 public snap_before;
    uint256 public snap_after;

    function run(address payable recipient) public {
        require(recipient != address(this));
        require(recipient != address(0));
        require(address(this).balance >= 10);
        snap_before = address(this).balance;
        Pay.send(recipient, 10);
        snap_after = address(this).balance;
        // After Stage 2 the delta is exactly 10, so asserting 42 is
        // refutable.
        assert(snap_before - snap_after == 42);
    }
}
