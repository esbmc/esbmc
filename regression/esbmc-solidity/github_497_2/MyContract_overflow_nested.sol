// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.26;

contract MyContract {
    uint8 x;
    uint8 y;
    uint8 z;
    uint8 sum;

    function func_overflow() external {
      x = 100;
      y = 240;
      z = 3;

      sum = x + y + z;
      assert(sum > 100);
    }
}
