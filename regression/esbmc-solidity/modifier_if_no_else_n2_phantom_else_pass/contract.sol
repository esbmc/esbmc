// SPDX-License-Identifier: MIT
// KNOWNBUG: `if (cond) _;` modifier (NO else) + 2-statement function body
// silently miscompiles. The splice walker erases the placeholder from
// op1() of the 2-op `ifthenelse(cond, _;)` and inserts N=2 body
// statements as siblings. Result: 3-op `ifthenelse(cond, stmt1, stmt2)`,
// which goto_convert interprets as a 3-op WITH-ELSE form:
// `if (cond) stmt1; else stmt2;`.
//
// The bug PROMOTES the second body statement into a phantom else-branch
// that runs for non-owner callers — the modifier's intent (no-else
// means "do nothing if !cond") is violated.
//
// Detector: constructor calls `setBoth()` (constructor is the owner of
// the contract, so the inner modifier `g` evaluates `cond=true`).
// Under correct semantics, both `x = 1` and `y = 2` execute → state
// (1, 2) and the post-call assertion holds. Under the bug, only the
// then-branch (`x = 1`) runs and `y = 2` is moved into a phantom else
// — owner-as-caller takes the then-branch, so y stays 0; assert fails.
//
// Pre-fix output: VERIFICATION FAILED (constructor's post-call
// assertion violated — only x is set).
// Post-fix output: VERIFICATION SUCCESSFUL.
//
// KNOWNBUG mode + regex `^VERIFICATION SUCCESSFUL$` silently passes
// today (regex does NOT match the FAILED output) and will flag for
// promotion to CORE once the splice fix lands.
//
// (User chose this n=2 phantom-else shape over the with-else+empty-body
// case because the latter is observationally invisible to ESBMC — both
// bug and fix leave state unchanged for empty-body functions.)
//
// Single-tx harness via the constructor avoids dispatch-loop k-induction
// convergence issues that affect a separate `probe()` style harness
// (the inductive step cannot relate `msg.sender == owner` across nondet
// dispatch iterations, returning UNKNOWN).
pragma solidity >=0.8.0;

contract C {
    address owner;
    uint256 x;
    uint256 y;

    constructor() {
        owner = msg.sender;
        setBoth();  // constructor is owner — modifier `g` permits entry
        assert(x == 1 && y == 2);
    }

    modifier g {
        if (msg.sender == owner) _;
    }

    function setBoth() public g {
        x = 1;
        y = 2;
    }
}
