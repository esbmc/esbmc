// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// #5 residual wall (a), ISOLATED from aqua's full machinery (minimal
// standalone sibling of cov_pilot_aqua2A_4lvl_addr_addr_bytes32_addr_uint256).
// A scalar (uint256-valued) 4-level nested-mapping WRITE in a
// calldata-array loop under branch-coverage k-induction. The
// structural abort is FIXED (solidity_convert_expr.cpp:4203) and the
// 2026-05-18 CVC5 auto-route fires (>=3-level nested mapping), so the
// base/forward phase resolves all 4 branch claims cleanly. Wall (a) —
// the while(nondet)dispatch() inductive step never converging within
// budget (reference_k_induction_budget_burn, solver-agnostic: Bitwuzla
// k=18 / CVC5 k=3) — is UNCHANGED and still a deferred future
// k-induction item.
//
// STAYS KNOWNBUG, and the reason is the regression harness, not esbmc:
// testing_tool.py STRIPS --timeout (UNSUPPORTED_OPTIONS) and bounds
// the run with its own ESBMC_REGRESS_TIMEOUT, then on TimeoutExpired
// returns stdout=None and self.fail()s UNCONDITIONALLY — the partial
// output is never regex-matched. So a perpetually-k-induction-timing-
// out test is an unconditional ctest failure in CORE *or* KNOWNBUG;
// KNOWNBUG is the honest mode (no spurious CORE-green claim).
//
// The 2026-05-18 "data even on UNKNOWN" esbmc fix DID land and is
// verified empirically: when esbmc is externally timeout-bounded
// (manual `--timeout`/SIGALRM, or `timeout(1)`/CI/the project-run
// orchestrator/SIGTERM), timeout_handler + term_handler
// (esbmc_parseoptions.cpp) emit the partial summary from an
// async-signal-safe snapshot before _exit:
//   [Coverage] Branches:4 Reached:2
//   Branch Coverage: 50% (partial: run terminated before verification
//   concluded)
// — a sound LOWER BOUND (2/4 edges witnessed; the other 2 were NOT
// proven unreachable). That is the user-facing "data even on UNKNOWN"
// requirement, delivered for the comparison pipeline; it is just NOT
// pinnable green through testing_tool's strip-+-unconditional-fail
// timeout architecture. See notes/Results/branch_cov/STAGE5_RESIDUAL_DIAG.md
// (Stage G). (The trivial `if`-only shape converges to 100% and does
// NOT reproduce this wall — the calldata-array loop is the amplifier.)
contract C {
  mapping(uint=>mapping(uint=>mapping(uint=>mapping(uint=>uint256)))) m;
  function f(uint a, uint b, uint[] calldata ks) external {
    for (uint i = 0; i < ks.length; i++) {
      uint v = m[a][b][i][ks[i]];
      require(v == ks.length);
      m[a][b][i][ks[i]] = 0xff;
    }
  }
}
