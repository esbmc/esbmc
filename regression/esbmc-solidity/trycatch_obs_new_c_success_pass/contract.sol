pragma solidity >=0.8.0;
contract C { uint256 v; constructor(uint256 x) { v = x; } }
contract H {
  function __ESBMC_reverted() internal returns (bool) {}
  function run(uint256 x) public {
    bool r; try new C(x) returns (C) { r = false; } catch { r = true; }
    assert(!r);                   // ctor always succeeds -> r false
  }
}
