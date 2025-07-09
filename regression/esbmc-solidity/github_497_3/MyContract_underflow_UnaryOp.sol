// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.26;

contract MyContract {
    uint8 x;

    function func_underflow() external {
      x = 1;
      --x;
      --x;
      assert(x < 5);
    }
}
