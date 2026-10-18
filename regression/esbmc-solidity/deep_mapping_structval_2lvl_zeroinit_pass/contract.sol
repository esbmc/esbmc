// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// KNOWNBUG: struct-valued nested mapping (K>=2) reads an unwritten slot as
// nondet instead of the EVM-mandated 0. Root cause: smt_conv.cpp:4754-4784
// (K>=2 array-of-struct init emits fresh unconstrained per-field arrays,
// "guarantees nothing"). Scalar-leaf mappings and K=1 struct mappings zero-init
// correctly; only nested struct-valued mappings regress. Expected SUCCESSFUL.
contract M2 {
  struct B { uint256 x; uint8 n; }
  mapping(address => mapping(address => B)) m;
  function c(address a) external view { assert(m[msg.sender][a].n == 0); }
}
