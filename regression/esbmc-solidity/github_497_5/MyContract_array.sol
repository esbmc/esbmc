// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.26;

contract MyContract {

    function func_array() external pure {
      uint8 i;
      uint8 x;
      uint8[2] memory a;

      if (x == 0)
      {
        a[i] = 0;
      }
      else
      {
        a[i+1] = 1;
      }

      assert(a[i+1] == 1);
    }
}
