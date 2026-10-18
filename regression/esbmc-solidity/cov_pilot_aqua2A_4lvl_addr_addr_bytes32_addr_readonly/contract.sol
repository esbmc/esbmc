// SPDX-License-Identifier: MIT
//
// CORE (flipped from KNOWNBUG 2026-05-15 by Stage 2C, commit
// 3d6d424b73).  Was pinned for the `bare smt_sort (id=4)` SMT-backend
// abort on a 4-level mapping-of-struct value (struct-of-arrays fix).
// Post-2C this read-only path now completes end-to-end and emits
// `Branch Coverage: 75%` (4 branches, 3 reached) — a genuine
// KNOWNBUG->CORE flip on a realistic nested mapping-of-struct shape.
// Sibling 4-level shapes that also *write* the nested mapping still
// abort at independent, pre-existing symex/IR walls (value_set
// base_type_eq / with2t is_array_type) and remain KNOWNBUG — see
// notes/Results/branch_cov/STAGE2C_FOLLOWUP_REPIN.md.
pragma solidity ^0.8.0;
struct Balance { uint248 amount; uint8 tokensCount; }
contract C {
    mapping(address => mapping(address => mapping(bytes32 => mapping(address => Balance)))) private _b;
    function rd(address app, bytes32 h, address[] calldata toks) external view returns (uint256 s) {
        for (uint256 i = 0; i < toks.length; i++) {
            Balance storage b = _b[msg.sender][app][h][toks[i]];
            require(b.tokensCount == toks.length);
            s += b.amount;
        }
    }
}
