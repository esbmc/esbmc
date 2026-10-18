// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// KNOWNBUG: same defect at 3 levels of struct-valued nesting (mirrors Aqua's
// _balances shape). Unwritten slot must read 0. See sibling
// deep_mapping_structval_2lvl_zeroinit_knownbug. Expected SUCCESSFUL.
contract M3 {
  struct B { uint256 x; uint8 n; }
  mapping(address => mapping(address => mapping(address => B))) m;
  function c(address a, address b) external view { assert(m[msg.sender][a][b].n == 0); }
}
