// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Race-mode TOD harness over a contract with a non-nullary constructor
// that takes a reference-type argument (`string memory`).  Before the
// ctor-args fix, the race harness emitted `new C()` with zero arguments
// — which fails to compile against ctors that require arguments — and
// the shadow-getter injection did not qualify string params with
// `memory`, tripping Solidity ≥0.5's data-location rule.
//
// Expected: bumpA/bumpB touch disjoint internal vars, so the harness
// verifies SUCCESSFUL after both ctor argument threading and the
// reference-type location qualifier are correct.
contract C {
    string private tag;
    uint256 internal a;
    uint256 internal b;

    constructor(string memory _tag) {
        tag = _tag;
    }

    function bumpA(uint256 n) public { a = a + n; }
    function bumpB(uint256 n) public { b = b + n; }
}
