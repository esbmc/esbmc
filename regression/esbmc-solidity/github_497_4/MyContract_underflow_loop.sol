// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.26;

contract MyContract {
    uint8 _x;

    function func_loop() external {
      _x = 2;
      for (uint8 i = 0; i < 3; ++i)
      {
        _x = _x - 1;
        assert(_x < 5);
      }
    }
}
