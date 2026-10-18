// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual of library_eoa_credit_unbound_pass.  After the fix,
// eoa.balance under --unbound correctly reflects the credit, so
// asserting it did NOT change is refutable.
contract C {
    function credit(address payable eoa) public {
        eoa.transfer(10);
    }

    function test(address payable eoa) public {
        require(eoa != address(0));
        require(eoa != address(this));
        require(eoa.balance == 100);
        credit(eoa);
        // Post-credit balance is 110, not 100.  Assertion must FAIL.
        assert(eoa.balance == 100);
    }
}
