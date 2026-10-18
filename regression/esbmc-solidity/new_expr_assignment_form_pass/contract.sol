// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

// Two contract instances created via the ASSIGNMENT form: the state vars are
// declared `C a; C b;` and assigned `new C()` inside the constructor (not a
// field initializer / local VariableDeclarationStatement).  Getters on such
// instances must execute the callee body precisely, not havoc to nondet.
contract C {
    uint256 public x = 7;
}

contract Harness {
    C a;
    C b;
    constructor() {
        a = new C();
        b = new C();
    }
    function check() public view {
        assert(a.x() == b.x());   // identical instances -> equal
        assert(a.x() == 7);       // precise body execution, not nondet
    }
}
