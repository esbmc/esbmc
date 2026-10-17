// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// #5 residual: non-vacuous FAIL dual of
// nested_mapping_write_3lvl_uint256_pass. The deep scalar write
// round-trips to v, so asserting v+1 must FAIL — proves the pass is
// not vacuously true.
contract C {
  mapping(uint=>mapping(uint=>mapping(uint=>uint256))) m;
  function f(uint i,uint j,uint k,uint v) public { m[i][j][k]=v; assert(m[i][j][k]==v+1); }
}
