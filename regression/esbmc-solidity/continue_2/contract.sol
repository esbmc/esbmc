// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;
contract MyContract {
  function test_continue() public
  {
    uint a = 0;
    for(uint i = 0; i < 3; i++)
    {
      if(i == 1)
      {
        continue;
      }

      a += 1;
    }
    assert(a == 2);
  }
}
