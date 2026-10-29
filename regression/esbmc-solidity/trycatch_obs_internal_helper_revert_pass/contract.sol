pragma solidity >=0.8.0;
contract C {
  function check(uint256 x) internal pure { require(x > 0, "pos"); }
  function f(uint256 x) external pure returns (uint256) { check(x); return x; }
}
contract H {
  C c;
  function __ESBMC_reverted() internal returns (bool) {}
  constructor() { c = new C(); }
  function run() public {
    bool r; try c.f(0) returns (uint256) { r = false; } catch { r = true; }
    assert(r);                    // helper reverts on x==0 -> catch
  }
}
