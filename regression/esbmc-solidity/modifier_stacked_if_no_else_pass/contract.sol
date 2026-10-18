// SPDX-License-Identifier: MIT
// Sanity test for stacked `if (cond) _;` modifiers under the
// splice_placeholders fix. Two modifiers `a` and `b`, both no-else,
// both with placeholder at op1() of an ifthenelse — applied to one
// function in order `a b`. The Solidity frontend processes modifiers
// in REVERSE order: iter1 = b (innermost), iter2 = a (outermost).
//
// Iter1: body_exprt = setBoth body (N=2). Modifier b's mod_body
//   contains `if (B) _;` — placeholder at fixed-arity parent. Post-fix
//   wrap: `if (B) { stmt1; stmt2 }`. After iter1, body_exprt is reset
//   to a single-call to b_aux (N=1).
// Iter2: body_exprt = { b_aux_call() } (N=1). Modifier a's mod_body
//   contains `if (A) _;` — placeholder at fixed-arity parent. Post-fix
//   wrap: `if (A) { b_aux_call() }` (semantically equivalent to legacy
//   flat splice for N=1).
//
// Calling f() then dispatches: a_aux → if (A) { b_aux() } → if (B) { x = 1; y = 2; }
// Owner is the constructor's deployer, both flags `A` and `B` are true,
// so the inner body runs; under fix, post-call x == 1 && y == 2.
pragma solidity >=0.8.0;

contract C {
    address owner;
    bool A;
    bool B;
    uint256 x;
    uint256 y;

    constructor() {
        owner = msg.sender;
        A = true;
        B = true;
        f();
        assert(x == 1 && y == 2);
    }

    modifier a {
        if (A && msg.sender == owner) _;
    }

    modifier b {
        if (B && msg.sender == owner) _;
    }

    function f() public a b {
        x = 1;
        y = 2;
    }
}
