// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// #5 residual: SCALAR-valued (uint256) 3-level nested-mapping WRITE
// round-trip. Companion to nested_mapping_write_3lvl_struct_pass (the
// existing dual is struct-only). Proves the 2026-05-15
// solidity_convert_expr.cpp:4203 fix is SOUND for the scalar leaf, not
// just struct. Solver pinned empirically (Stage P.4).
contract C {
  mapping(uint=>mapping(uint=>mapping(uint=>uint256))) m;
  function f(uint i,uint j,uint k,uint v) public { m[i][j][k]=v; assert(m[i][j][k]==v); }
}
