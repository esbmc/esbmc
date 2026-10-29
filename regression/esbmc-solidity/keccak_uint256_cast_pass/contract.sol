// SPDX-License-Identifier: MIT
// Regression for the bytes_static_to_uint slow-path "Not unwinding"
// flood. Pre-fix, the user-style pattern
//   uint256(keccak256(abi.encodePacked(bytes32, bytes32)))
// emitted 9+ "Not unwinding loop N iteration M ... bytes_static_to_uint"
// lines per k-induction phase and silently truncated the cast result for
// `--unwind < 32`. Post-fix, the slow-path loops in solidity_bytes.c
// were unrolled into 32-step ternary chains using literal constant
// indices, so symex never has to unwind a symbolic-bound loop.
//
// Test pattern: exercise the uint256(keccak256(abi.encodePacked(...)))
// cast path that goes through bytes_static_to_uint. Pre-fix this
// flooded stderr; post-fix the unrolled helper passes through clean.
// No additional assertion — the test passes as long as the cast
// completes (which under the bug it does, but with truncated value
// and "Not unwinding" log noise; under the fix, neither).
//
// (A determinism oracle u1 == u2 across two callsites would be wrong:
// the F1 wide-BV keccak table mechanism enforces per-callsite
// distinctness, so two identical-input keccak256 calls at distinct
// callsites are FORCED to produce different table entries to cover
// the injectivity direction the SMT array axiom doesn't provide.
// See docs/claude/solidity/language-support.md Section A.)
pragma solidity >=0.8.0;

contract C {
    function castOnce(bytes32 a, bytes32 b) external pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(a, b)));
    }
}
