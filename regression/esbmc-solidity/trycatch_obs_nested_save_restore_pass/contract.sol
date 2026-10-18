pragma solidity >=0.8.0;
contract D { function bad(uint256) external pure { require(false, "d"); } }
contract C {
  D d;
  function __ESBMC_reverted() internal returns (bool) {}
  constructor(D _d) { d = _d; }
  function f(uint256 x) external returns (uint256) {
    bool ri; try d.bad(x) { ri = false; } catch { ri = true; }  // marks then restores flag
    return x;                     // c.f does NOT revert
  }
}
contract H {
  C c; D d;
  function __ESBMC_reverted() internal returns (bool) {}
  constructor() { d = new D(); c = new C(d); }
  function run(uint256 x) public {
    bool ro; try c.f(x) returns (uint256) { ro = false; } catch { ro = true; }
    assert(!ro);                  // inner d.bad revert must NOT leak to outer
  }
}
