// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// #5 residual: non-vacuous FAIL dual of
// nested_mapping_write_4lvl_uint256_pass.
contract C {
  mapping(uint=>mapping(uint=>mapping(uint=>mapping(uint=>uint256)))) m;
  function f(uint i,uint j,uint k,uint l,uint v) public { m[i][j][k][l]=v; assert(m[i][j][k][l]==v+1); }
}
