// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// CORE: the struct-valued nested-mapping array round-trips reads/writes
// correctly; only the *default* of an unwritten slot is wrong. Writing then
// reading the same slot must return the written value. Guards against a fix
// that zeroes written slots (over-constraint).
contract W3 {
  struct B { uint256 x; uint8 n; }
  mapping(address => mapping(address => mapping(address => B))) m;
  function c(address a, address b) external {
    m[msg.sender][a][b].n = 7;
    assert(m[msg.sender][a][b].n == 7);
  }
}
