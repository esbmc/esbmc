// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// #5 wall (b) — NON-VACUITY dual of
// nested_mapping_write_uint256_autoroute_cvc5_pass.  Same 4-level
// scalar nested-mapping shape, same NO-explicit-solver setup (so the
// esbmc_parseoptions.cpp `t_mapping$_`x3 deep-mapping detector fires
// and auto-selects plain CVC5).  The post-write assert is deliberately
// WRONG (`== v + 1`).  This proves the auto-routed CVC5 path is
// non-vacuous: it does NOT rubber-stamp SUCCESSFUL — it still finds a
// genuine counterexample on the scalar deep-nested mapping write that
// Bitwuzla could not even encode (const-array-eq abort).  Pair with
// the _pass sibling this documents the routing fix is both sound
// (real violations still caught) and complete (the legitimate
// round-trip verifies) on the previously-default-solver-crashing
// shape.  Override-the-route control is the 5 explicit-`--cvc5`
// nested_mapping_write_{3,4}lvl_uint256_* duals.
contract C {
  mapping(uint=>mapping(uint=>mapping(uint=>mapping(uint=>uint256)))) m;
  function f(uint i,uint j,uint k,uint l,uint v) public { m[i][j][k][l]=v; assert(m[i][j][k][l]==v+1); }
}
