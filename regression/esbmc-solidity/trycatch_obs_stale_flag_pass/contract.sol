pragma solidity >=0.8.0;
contract A { function g(uint256) external pure { require(false, "a"); } }
contract B { function f(uint256 x) external pure returns (uint256) { return x; } }
contract H {
  A a; B b;
  function __ESBMC_reverted() internal returns (bool) {}
  constructor() { a = new A(); b = new B(); }
  function run(uint256 x) public {
    a.g(x);                       // reverts -> global flag set
    bool r; try b.f(x) returns (uint256) { r = false; } catch { r = true; }
    assert(!r);                   // pre-call clear => r false despite stale flag
  }
}
