// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;
contract MyContract {
  uint a = 0;
  function test_continue() public
  {
    for(uint i = 0; i < 10; i++)
    {
      if(i == 1)
      {
        continue;
      }

      a += 1;
    }
    assert(a == 10);
  }
}