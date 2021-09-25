// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.26;

contract MyContract {
    uint8 x;
    uint8 sum;

    function nondet() public pure returns(uint8)
    {
      uint8 i;
      return i;
    }

    //function __ESBMC_assume(bool) internal pure { }
    function __VERIFIER_assume(bool) internal pure { }

    function func_sat() external {
      x = 0;
      uint8 y = nondet();
      sum = x + y;

      // C : Add additional constraints here
      __VERIFIER_assume(y < 255);
      __VERIFIER_assume(y > 220);
      __VERIFIER_assume(y != 224); // 224 = 16 * 14;

      // P : Properties we want to check
      assert(sum % 16 != 0);
    }
}


      //__ESBMC_assume(y < 255);
      //__ESBMC_assume(y > 220);
      //__ESBMC_assume(y != 224); // 224 = 16 * 14;
      //__ESBMC_assume(y != 240); // 240 = 16 * 15;
/*
C  = [
       y=0             /\
       z=nondet()      /\
       sum = y+z       /\
       sum1 = sum % 16 /\
       _z != 224      /\
       _z != 240      /\
       z < 255 /\ z > 220
     ]

P = [ sum1 != 0 ]
~P = [ sum1 == 0 ]

Find a counterexample that satisfies C /\ ~P

satisfiable?

*/

/*
  __ESBMC_assume(_z < 255);
  __ESBMC_assume(_z > 220);
  __ESBMC_assume(_z != 240); // 240 = 16 * 15;
  __ESBMC_assume(_z != 224); // 224 = 16 * 14;
  assert(sum % 16 != 0);
*/
