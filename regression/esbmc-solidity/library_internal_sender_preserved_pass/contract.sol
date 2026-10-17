// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Complement to library_external_call_sender_overapprox_fail.  Inside
// a library body (BEFORE any external call from that library), real
// Solidity semantics require msg.sender to be inherited from the
// calling contract's own msg.sender — library internal calls do
// NOT switch context.  ESBMC preserves this by NOT emitting a
// contract-method wrapper around the `Lib.record(...)` call itself:
// A.invoke() calls Lib.record() as a plain function call, so
// msg_sender retains whatever the wrapper on `a.invoke()` (applied
// at Harness.test's call site) assigned.
//
// Regression behavior: both the library body and the contract method
// that called it should observe the SAME msg.sender.  The library
// reads it into a local `sender`, returns it, and A.invoke() cross-
// checks against its own `msg.sender` — identical because no wrapper
// sat between them.  Discharged statically because symex sees the
// same symbol on both sides (no nondet intervenes).
library Lib {
    function readSender() internal view returns (address) {
        return msg.sender;
    }
}

contract A {
    function invoke() public view returns (address, address) {
        address libSide = Lib.readSender();
        return (libSide, msg.sender);
    }

    function check() public view {
        (address lib, address self) = invoke();
        assert(lib == self);
    }
}

contract Harness {
    function test() public {
        A a = new A();
        a.check();
    }
}
