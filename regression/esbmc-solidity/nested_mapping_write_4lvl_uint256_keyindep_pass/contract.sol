// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// #5 residual: depth-4 SCALAR key-independence. Writing a different
// innermost key (l2 != l) must NOT clobber m[i][j][k][l] — proves the
// post-4203 deep scalar nested-mapping slots are disjoint per key (no
// slot aliasing in the under-nested-index fix). --cvc5 (Bitwuzla
// const-array-eq, see the _knownbug sibling).
contract C {
  mapping(uint=>mapping(uint=>mapping(uint=>mapping(uint=>uint256)))) m;
  function f(uint i,uint j,uint k,uint l,uint l2,uint v) public {
    require(l2 != l);
    m[i][j][k][l]=v;
    m[i][j][k][l2]=0;
    assert(m[i][j][k][l]==v);
  }
}
