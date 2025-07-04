// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.26;

contract MyContract {
    uint8 x;
    uint8 sum;

    function nondet() public pure returns(uint8)
    {
      uint8 i;
      return i;
    }

    function __ESBMC_assume(bool) internal pure { }

    function func_sat() external {
      x = 0;
      uint8 y = nondet();
      sum = x + y;

      // C : Add additional constraints here
      __ESBMC_assume(y < 255);
      __ESBMC_assume(y > 220);

      // P : Properties we want to check
      assert(sum % 16 != 0);
    }
}
