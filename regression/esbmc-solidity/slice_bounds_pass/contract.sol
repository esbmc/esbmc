// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #14: IndexRangeAccess (b[s:e]) is currently a fresh nondet
// detached from base, with NO constraint `s <= e <= base.length`. SMT
// can satisfy the slice with s > e or e > b.length — both are reverts
// in real EVM. After the fix, the assume guards make `s > e` infeasible
// when the slice is reached.
contract S {
    function check(bytes calldata b, uint s, uint e) external pure {
        // Reach the slice — this should constrain the path to s <= e.
        bytes calldata slice = b[s:e];
        (slice);  // suppress unused-variable warning
        // Post-slice, the path must satisfy `s <= e`. Pre-fix: not
        // constrained, SMT picks s > e and the assertion fails. Post-fix:
        // assume guard prunes s > e paths, assertion holds.
        assert(s <= e);
    }
}
