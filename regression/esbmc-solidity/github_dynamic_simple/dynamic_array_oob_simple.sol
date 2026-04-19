// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.26;

contract MyContract {
  function dyn_array_oob_simple(uint8 n) public pure {
    uint8[] memory a = new uint8[](n);
    a[0] = 100;
    assert(a[0] == 100);
  }
}
