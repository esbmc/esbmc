// SPDX-License-Identifier: MIT
// Regression for the multi-modifier param-drop IR crash.
// Pre-fix, the intermediate-modifier-call branch in
// solidity_convert_modifier.cpp:1352-1388 built each chained call as
// [this, mod_args], omitting the wrapped function's formal parameters
// even though the next wrapper's signature (built at lines 1077-1106)
// declared them. The final-call branch at lines 1389-1431 forwarded
// the params correctly via the loop at 1402-1422 — the intermediate
// branch was the only branch missing the loop. ESBMC then emitted a
// sourceless "Violated property: function call: not enough arguments
// / VERIFICATION FAILED / Bug found (k = 1)" at base case k=1,
// blocking detection of any real property in the contract.
//
// Existing modifier_3 / modifier_4 use a no-parameter wrapped
// function `function func2() public check check2 { ... }`, so the
// missing parameter loop iterated zero times and the bug stayed
// silent. This regression covers the parameterised path.
//
// Post-fix: the missing loop is mirrored from the final-call branch,
// inserted between the `this_ptr` push and the next-modifier scope
// switch. The wrapped-function parameter symbols, registered under
// the outer wrapper's scope by the signature-construction loop at
// lines 1087-1098, are pushed onto the call before the modifier
// invocation arguments. The assertion `x == before + a` is a true
// invariant of the body `x = x + a`, so VERIFICATION SUCCESSFUL is
// the correct verdict.
//
// Originally pinned as KNOWNBUG (commit 12432184d1); flipped to CORE
// in the wrapped-params-forwarding fix at commit dcfcb34a16.
pragma solidity >=0.8.0;
contract C {
    uint256 public x;
    modifier m1() { _; }
    modifier m2() { _; }
    function f(uint256 a) public m1 m2 {
        uint256 before = x;
        x = x + a;
        assert(x == before + a);
    }
}
