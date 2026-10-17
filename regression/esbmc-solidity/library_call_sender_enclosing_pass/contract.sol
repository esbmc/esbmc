// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// KNOWNBUG: When a library body performs an external call, the
// callee should observe `msg.sender == address(A)` where A is the
// ENCLOSING CONTRACT that invoked the library.  Our model currently
// emits `msg_sender = NONDET(uint160)` at library-scope external
// calls (D's over-approximation in
// src/solidity-frontend/solidity_convert_contract.cpp), because the
// enclosing contract is run-time data that the frontend can't pin.
//
// Once the fix is in (threading caller's `this` through library
// invocations), the callee observes A's address deterministically
// and the assertion `t.lastSender() == address(a)` holds, flipping
// to VERIFICATION SUCCESSFUL.
contract Target {
    address public lastSender;
    function ping() public { lastSender = msg.sender; }
}

library Nudge {
    function nudge(Target t) internal {
        t.ping();
    }
}

contract A {
    Target public target;
    function init(Target t) public { target = t; }
    function go() public { Nudge.nudge(target); }
}

contract Harness {
    function test() public {
        Target t = new Target();
        A a = new A();
        a.init(t);
        a.go();
        // Correct: callee saw msg.sender == A's address.
        // Current bug: NONDET msg.sender can legally be any value.
        assert(t.lastSender() == address(a));
    }
}
