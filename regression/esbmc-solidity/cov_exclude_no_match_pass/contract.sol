// Item 5-d (--coverage-exclude-contract) regression source. Shared by:
//   cov_exclude_whole_unit_pass  : whole-unit + exclude OZLib => Branches 2
//     (OZLib own 4 edges leave BOTH denominator and numerator).
//   cov_exclude_default_noop_pass: default --contract C + exclude => Branches 2
//     (== no-flag baseline; exclude is a no-op in semantics-A default mode).
//   cov_exclude_no_match_pass    : whole-unit + non-matching name => Branches 6
//     (exclusion fires ONLY on exact sol_decl_contract match).
// Dual: whole_unit_pass(2) vs no_match_pass(6) differ ONLY by whether the
// excluded name matches, so a numerator-only regression (denominator still
// counts OZLib) flips Branches 2->6 and ctest catches it.

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OZLib {
    uint256 internal _v;
    function ozStep(uint256 x) internal {
        if (x > 10) { _v = x; } else { _v = 0; }
        if (x % 2 == 0) { _v = _v + 1; }
    }
}

contract C is OZLib {
    function run(uint256 a) public {
        ozStep(a);
        if (a > 100) { _v = a * 2; } else { _v = a; }
    }
}
