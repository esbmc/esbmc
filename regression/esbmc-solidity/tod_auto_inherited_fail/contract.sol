// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Auto-discovered TOD pair across an inheritance chain.  Base contract
// declares an internal state var AND a function that writes it; the
// derived contract declares only the reader.  Without the inheritance
// walk, --tod-race-check=auto scanned only the derived contract's own
// body and missed the writer entirely, dropping leaf-contract recall
// on TransRacer-style benchmarks (Finding 2 of the SolidiFi batch).
//
// With the fix, auto-mode picks up `setV` (inherited from Base) and
// `bumpV` (declared on Leaf), both touching `v`, and the pair is
// reported as a race.  Expected verdict: VERIFICATION FAILED — the
// two orderings on the same initial state produce different `v`.
contract Base {
    uint256 internal v;
    function setV(uint256 n) public { v = n; }
}

contract Leaf is Base {
    function bumpV(uint256 n) public { v = v + n; }
}
