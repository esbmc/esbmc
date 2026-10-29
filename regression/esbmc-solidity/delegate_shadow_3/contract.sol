// SPDX-License-Identifier: GPL-3.0
// KNOWNBUG: the delegatecall re-enters ProxyMis without bound. The inductive
// step answers the re-entries past its bound with a havoc of what a dispatch
// writes, `x` among them, so the innermost unfolded `assert(x == 7)` sees an
// arbitrary `x`. Proving it needs the invariant `x == 0 || x == 7`, which
// k-induction does not infer. The step used to cut the re-entries off
// instead, which is unsound (see reentry_kinduction_deep_fail).
pragma solidity >=0.8.0;

// Layout mismatch: Logic accesses state var `y` which Proxy does not
// define. The delegate-shadow fast path must reject this case at
// compile-time and fall back to the generic $delegatecall#0 helper,
// which returns false without touching Proxy state. Proxy.x must
// remain at its initial value.
contract LogicMis {
    uint256 public y;
    function setY(uint256 v) public { y = v; }
}

contract ProxyMis {
    uint256 public x;
    // Note: no `y` field — shadow dispatch for setY(uint256) must fail
    // compatibility validation.

    function test() public {
        x = 7;
        address(this).delegatecall(
            abi.encodeWithSignature("setY(uint256)", uint256(99))
        );
        // Nothing Proxy sees should have changed.
        assert(x == 7);
    }
}
