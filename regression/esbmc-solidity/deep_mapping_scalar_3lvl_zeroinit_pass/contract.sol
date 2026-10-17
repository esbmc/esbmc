// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// CORE: scalar-leaf nested mapping zero-inits correctly at ANY depth. This
// pins the boundary — the bug is struct-leaf, NOT depth (depth-3 scalar passes,
// depth-2 struct fails). Guards against a fix that over-broadly changes scalar
// mappings.
contract S3 {
  mapping(address => mapping(address => mapping(address => uint256))) m;
  function c(address a, address b) external view { assert(m[msg.sender][a][b] == 0); }
}
