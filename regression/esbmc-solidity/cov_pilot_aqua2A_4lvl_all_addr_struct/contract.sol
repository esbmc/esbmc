// SPDX-License-Identifier: MIT
//
// CORE (flipped from KNOWNBUG 2026-05-15 by the deep-nested-mapping
// WRITE fix in src/solidity-frontend/solidity_convert_expr.cpp:4203:
// the nested mapping-access branch now types the intermediate
// index_exprt with array.type().subtype() instead of the under-nested
// get_type_description `t` (mirrors the direct-access fast path :4174).
// Previously: bare smt_sort (Stage 0/1) -> after Stage 2C, symex/IR
// abort (irep2_expr.cpp:366 / value_set.cpp:1258) on the >=3-level
// nested-mapping storage-ref write.  Now completes: Branch Coverage
// 75% (4 branches, 3 reached).  See
// notes/Results/branch_cov/STAGE2C_FOLLOWUP_DIAG.md.
pragma solidity ^0.8.0;
struct Balance { uint248 amount; uint8 tokensCount; }
contract C {
    mapping(address => mapping(address => mapping(address => mapping(address => Balance)))) private _b;
    function dock(address app, address h, address[] calldata toks) external {
        for (uint256 i = 0; i < toks.length; i++) {
            Balance storage b = _b[msg.sender][app][h][toks[i]];
            require(b.tokensCount == toks.length);
            b.tokensCount = 0xff;
        }
    }
}
