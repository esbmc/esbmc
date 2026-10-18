pragma solidity >=0.8.0;
contract C {
  modifier onlyPos(uint256 x) { require(x > 0, "pos"); _; }
  function f(uint256 x) external onlyPos(x) returns (uint256) { return x; }
}
contract H {
  C c;
  function __ESBMC_reverted() internal returns (bool) {}
  constructor() { c = new C(); }
  function run() public {
    bool r; try c.f(0) returns (uint256) { r = false; } catch { r = true; }
    assert(r);                    // modifier reverts on x==0 -> catch
  }
}
