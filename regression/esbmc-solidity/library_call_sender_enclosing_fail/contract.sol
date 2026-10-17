// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual of library_call_sender_enclosing_pass.  msg.sender is
// now deterministically the enclosing contract's address inside the
// library's external call, so asserting the callee saw a specific
// OTHER address is refutable.
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
        // Correct: lastSender == address(a).  Asserting it equals
        // address(0) (a specific different value) must fail.
        assert(t.lastSender() == address(0));
    }
}
