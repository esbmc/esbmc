// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// #5 wall (b) — FIXED & FLIPPED KNOWNBUG->CORE 2026-05-18 by the
// deep-nested-mapping CVC5 auto-route in esbmc_parseoptions.cpp.
// No explicit solver: pre-fix this fell to the DEFAULT solver
// (Bitwuzla), where the scalar deep nested-mapping write made
// IS-havoc compose an asymmetric `(= ca freshsym)` over a CONST_ARRAY
// infinite mapping array and Bitwuzla aborted with "Equality over
// constant arrays not fully supported yet" (upstream
// array_solver.cpp:225-241). The new detector recognises the
// `t_mapping$_`×3 typeIdentifier and auto-selects plain CVC5, which
// verifies cleanly: `VERIFICATION SUCCESSFUL`. This is the
// end-to-end proof that the default-solver crash on a scalar >=3-level
// nested-mapping write is closed (systemic, deterministic). The
// upstream Bitwuzla const-array-eq limitation itself is unchanged and
// out of scope (tracked in memory reference_bitwuzla_const_array_eq_trigger).
contract C {
  mapping(uint=>mapping(uint=>mapping(uint=>mapping(uint=>uint256)))) m;
  function f(uint i,uint j,uint k,uint l,uint v) public { m[i][j][k][l]=v; assert(m[i][j][k][l]==v); }
}
