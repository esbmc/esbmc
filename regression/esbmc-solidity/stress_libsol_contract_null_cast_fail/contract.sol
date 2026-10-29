// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to stress_libsol_contract_null_cast_pass: validates bug-finding
// after the contract-null-cast fix. The function asserts that the
// state-var pointer is *non-null*, but the freshly initialised state
// has p == Pool(0), so the assertion must be violated and report
// VERIFICATION FAILED.

contract Pool { uint256 public x; }

contract Factory {
    Pool public p;

    function check() public view {
        assert(p != Pool(address(0)));
    }
}
