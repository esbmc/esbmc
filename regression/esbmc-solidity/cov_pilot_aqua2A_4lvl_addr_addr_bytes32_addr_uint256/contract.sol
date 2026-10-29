// SPDX-License-Identifier: MIT
//
// KNOWNBUG (canonical #5 pin). The deep-nested-mapping WRITE symex/IR
// abort (irep2_expr.cpp:366) IS fixed (solidity_convert_expr.cpp:4203);
// this pilot no longer aborts.  Two further walls were diagnosed
// (STAGE5_RESIDUAL_DIAG.md): (b) bitwuzla const-array-eq under
// assertion BMC, and (a) k-induction non-convergence under coverage.
// The 2026-05-18 CVC5 auto-route (esbmc_parseoptions.cpp deep-mapping
// detector) CLOSES wall (b) end-to-end (see
// nested_mapping_write_uint256_autoroute_cvc5_pass).
//
// Wall (a) — the Solidity while(nondet)dispatch() inductive step never
// converging within budget (reference_k_induction_budget_burn,
// solver-agnostic: Bitwuzla k=18 / CVC5 k=3) — is UNCHANGED and still
// a deferred future k-induction item.
//
// STAYS KNOWNBUG because of the harness, not esbmc: testing_tool.py
// STRIPS --timeout and on its own-timeout returns stdout=None +
// self.fail()s UNCONDITIONALLY (the partial output is never
// regex-matched), so a perpetually-timing-out test is an unconditional
// ctest failure in CORE or KNOWNBUG alike — KNOWNBUG is the honest
// mode.  The 2026-05-18 "data even on UNKNOWN" esbmc fix DID land
// (timeout_handler SIGALRM + new term_handler SIGTERM/SIGINT, both
// emitting the partial summary from an async-signal-safe snapshot
// before _exit) and is verified empirically: externally
// timeout-bounded (manual --timeout, `timeout(1)`/CI, the project-run
// orchestrator) this pin emits `Branch Coverage: 50% (partial: ...)`
// — a sound LOWER BOUND (CVC5 clears base/forward; 2/4 edges
// witnessed; the other 2 NOT proven unreachable).  That delivers the
// user-facing "data even on UNKNOWN" requirement for the comparison
// pipeline; it is simply not pinnable green through testing_tool's
// strip-+-unconditional-fail timeout design.
// See notes/Results/branch_cov/STAGE5_RESIDUAL_DIAG.md (Stage G).
pragma solidity ^0.8.0;
contract C {
    mapping(address => mapping(address => mapping(bytes32 => mapping(address => uint256)))) private _b;
    function dock(address app, bytes32 h, address[] calldata toks) external {
        for (uint256 i = 0; i < toks.length; i++) {
            uint256 v = _b[msg.sender][app][h][toks[i]];
            require(v == toks.length);
            _b[msg.sender][app][h][toks[i]] = 0xff;
        }
    }
}
