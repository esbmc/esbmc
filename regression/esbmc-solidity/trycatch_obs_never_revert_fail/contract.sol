pragma solidity >=0.8.0;
contract C { function f(uint256 x) external pure returns (uint256) { return x; } }
contract InvMutTest {
  C c; function __ESBMC_reverted() internal returns (bool) {}
  constructor() { c = new C(); }
  function run(uint256 x) public { bool r; try c.f(x) returns (uint256) { r=false; } catch { r=true; } assert(r); }
}
