// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract C {
  function f() public pure {
    uint x;
    uint y;
    if (++x < 3)
      y = 1;
 
    assert(x == 1);
  }
}
