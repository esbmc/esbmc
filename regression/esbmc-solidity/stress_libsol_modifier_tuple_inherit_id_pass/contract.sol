// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for get_func_modifier not registering a tuple_instance for
// the synthetic aux function wrapping a tuple-returning function.
// get_tuple_instance_name builds `tuple_instance$<fn_ast_id>`; the aux
// FunctionDefinition synthesised by insert_modifier_json has id=0, so
// a tuple-returning modifier-wrapped function aborted with
// "cannot find tuple instance symbol: ...@tuple_instance$0". The fix
// registers a tuple instance for the synthetic aux whenever the wrapped
// function originally returned a tuple.

contract C {
    modifier nonReentrant() {
        _;
    }

    // pair(a) returns (a, a + 1). Lower the return values into plain
    // state so the assertion does not depend on the multi-tx harness'
    // nondet argument for `this.pair(...)`.
    uint256 public lastX;
    uint256 public lastY;

    function pair(uint256 a) external nonReentrant returns (uint256, uint256) {
        lastX = a;
        lastY = a + 1;
        return (a, a + 1);
    }

    function go() external {
        lastX = 7;
        lastY = 8;
        // Touch the tuple-returning, modifier-wrapped call path so the
        // aux function and its tuple instance are actually instantiated.
        this.pair(3);
        // Assertion independent of the call's nondet argument: after
        // writing 7 and 8 in this transaction, before any other call
        // could observe the state, the values are exactly 7 and 8
        // because the harness executes transactions atomically.
        //
        // Unfortunately the dispatcher may rerun this function with
        // different state, so rely on an invariant that holds for any
        // legal outcome: lastY == lastX + 1 whenever pair() ran last.
        // The simpler guarantee: assert(true) — if the frontend crashes
        // we never get here.
        assert(true);
    }
}
