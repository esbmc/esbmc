// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;
contract C {
  function _ESBMC_Main_fake(uint n) external returns (uint) {
    if (n == 0) return 0;
    return this._ESBMC_Main_fake(n - 1);
  }
  function check() public {
    assert(this._ESBMC_Main_fake(3) != 0);
  }
}
