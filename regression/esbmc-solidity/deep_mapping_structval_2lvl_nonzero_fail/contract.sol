// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// CORE (negative dual of deep_mapping_structval_2lvl_zeroinit): an unwritten
// struct-valued nested-mapping slot is 0, so asserting it is nonzero must FAIL.
contract M2N {
  struct B { uint256 x; uint8 n; }
  mapping(address => mapping(address => B)) m;
  function c(address a) external view { assert(m[msg.sender][a].n != 0); }
}
