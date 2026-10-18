// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// #5 residual: SCALAR-valued (uint256) 4-level nested-mapping WRITE
// round-trip. This is the exact shape of the canonical residual pin
// cov_pilot_aqua2A_4lvl_addr_addr_bytes32_addr_uint256, isolated and
// proven SOUND under --cvc5 (Bitwuzla trips const-array-eq; see
// nested_mapping_write_uint256_bitwuzla_constarrayeq_knownbug). cvc5
// handling the scalar deep write correctly is the evidence the fix
// lever is solver-routing, not symex/IR.
contract C {
  mapping(uint=>mapping(uint=>mapping(uint=>mapping(uint=>uint256)))) m;
  function f(uint i,uint j,uint k,uint l,uint v) public { m[i][j][k][l]=v; assert(m[i][j][k][l]==v); }
}
