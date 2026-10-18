// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract C {
  struct S { uint256 a; uint256 b; }
  mapping(uint=>mapping(uint=>mapping(uint=>S))) m;
  function f(uint i,uint j,uint k,uint v) public { m[i][j][k].a=v; assert(m[i][j][k].a==v); }
}
