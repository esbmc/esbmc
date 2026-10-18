// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual of library_low_level_call_compile: asserts that the
// boolean `success` returned from a library-scoped `.call()` is
// observable-as-nondet.  Under the root-cause fix (populate library
// `$call#0` with the dispatch ladder that returns a path-conditional
// bool), `success` may be either true (address matched a tracked
// contract, Nondet_Extcall invoked) or false (address matched no
// contract, falls through to `return false`).  `assert(success)`
// therefore must be refutable.
//
// This is the regression guard that prevents silent regression to the
// earlier workaround, which emitted a pure nondet-bool return body
// (the assert would have the same pass/fail shape — not informative)
// vs. the dispatch ladder (success depends on the concrete address
// matching a known contract, which with an arbitrary param address
// symex cannot prove).
contract Target {
    function setFlag() public {}
}

library Lib {
    function poke(address addr, bytes memory data) internal returns (bool) {
        (bool ok, ) = addr.call(data);
        return ok;
    }
}

contract Caller {
    function test(address a, bytes memory d) public {
        bool ok = Lib.poke(a, d);
        assert(ok);
    }
}
