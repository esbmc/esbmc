#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_convert_class.h>
#include <goto-programs/loop_unroll.h>
#include <langapi/language_util.h>
#include <map>
#include <unordered_set>
#include <array>
#include <atomic>
#include <map>
#include <mutex>

class goto_coveraget
{
public:
  explicit goto_coveraget(const namespacet &ns, goto_functionst &goto_functions)
    : ns(ns), goto_functions(goto_functions)
  {
    target_num = -1;
  };
  explicit goto_coveraget(
    const namespacet &ns,
    goto_functionst &goto_functions,
    const std::string filename)
    : ns(ns), goto_functions(goto_functions), filename(filename)
  {
    target_num = -1;
  };
  void assertion_coverage();
  void branch_coverage();
  void branch_function_coverage();

  // k-path coverage: at each branch, emit goals for every combination of the
  // last (n-1) branch directions and the current direction (Williams et al.,
  // EDCC 2005). A goal is `assert(!witness)`; SAT means the path is reachable
  // and the goal is marked covered by multi_property_check, mirroring the
  // branch_coverage convention. The structural AND-chain preserves SSA merge
  // friendliness up to depth_cap; past depth_cap the witness is too deep to
  // hand to the solver and is dropped (a Phase-2 ghost-flag fallback is
  // tracked in #4325). Goal count is bounded by max_goals per function.
  void k_path_coverage();

  // Solidity complete-path coverage (paper: entry->exit path coverage for
  // test generation). Unlike k_path_coverage (prefix witnesses asserted AT
  // the branch), this snapshots each decision's guard VALUE into a ghost
  // symbol on the decision edge (ASSIGN c_i = guard, via cov_context), then
  // asserts assert(!(c_1==d_1 && ... && c_k==d_k)) on each edge reaching
  // END_FUNCTION for every enumerated complete decision sequence. A probe
  // is falsified => that complete path is feasible => multi_property emits
  // its CE. The per-path enc(pi) is encoded into the claim comment so
  // claim_sig stays unique per path (bmc.cpp:2000 is otherwise unsound).
  // Slice 1: loop-free functions (acyclic DFS, scalar snapshots). Loops
  // (occurrence-indexed snapshot slots) are Slice 2. Requires cov_context.
  void solidity_path_coverage();

  void insert_assert(
    goto_programt &goto_program,
    goto_programt::targett &it,
    const expr2tc &guard);

  // customize comment
  void insert_assert(
    goto_programt &goto_program,
    goto_programt::targett &it,
    const expr2tc &guard,
    const std::string &idf);

  // replace every assertion to a specific guard
  void replace_all_asserts_to_guard(
    const expr2tc &guard,
    bool is_instrumentation = false);
  // replace an assertion to a specific guard
  void replace_assert_to_guard(
    const expr2tc &guard,
    goto_programt::instructiont::targett &it,
    bool is_instrumentation);

  // convert assert(cond) to assume(cond) to preserve path constraints
  void replace_assert_to_assume(goto_programt::instructiont::targett &it);
  void replace_all_asserts_to_assume();

  // convert assert(cond) to assert(!cond)
  void negating_asserts(const std::string &tgt_fname);

  // condition cov
  void condition_coverage();
  expr2tc gen_not_eq_expr(const expr2tc &lhs, const expr2tc &rhs);
  expr2tc gen_and_expr(const expr2tc &lhs, const expr2tc &rhs);
  expr2tc gen_not_expr(const expr2tc &expr) const;
  int get_total_instrument() const;
  int get_total_assert_instance() const;
  std::set<std::pair<std::string, std::string>> get_total_cond_assert() const;
  std::string get_filename_from_path(std::string path);
  void set_target(const std::string &_tgt);
  bool is_target_func(const irep_idt &f, const std::string &tgt_name) const;
  bool
  filter(const irep_idt &func_name, const goto_programt &goto_program) const;

  // total numbers of instrumentation
  static size_t total_assert;
  static size_t total_assert_ins;
  static std::set<std::pair<std::string, std::string>> total_cond;
  static size_t total_branch;
  static size_t total_func_branch;
  static size_t total_kpath;
  // |spanning_set| under Marré-Bertolino subsumption (issue #4335 PR1).
  // Equals total_kpath when no goal is subsumed by another. Used as the
  // denominator of the k-path coverage percentage to drop the lower
  // bound contribution of redundant subsumed goals.
  static size_t total_kpath_spanning;
  // (msg, loc) pairs whose every emission is non-maximal. JSON report
  // marks these as "spanning-set-redundant" and they are excluded from
  // the spanning-set denominator.
  static std::set<std::pair<std::string, std::string>>
    k_path_spanning_redundant;
  // all instrumented claims (condition, location) for JSON report.
  // For branch coverage this is the *static universe* (every in-scope
  // decision edge, built with NO covered-set skip applied) so the
  // denominator never shrinks when edges are skipped (Item 2c).
  static std::set<std::pair<std::string, std::string>> all_claims;
  // Guard text of the opposite direction for each claim in all_claims, keyed
  // the same way. A probe assert(c) left unviolated proves !c infeasible, so
  // --dead-code-check names this rather than the claim's own comment. Derived
  // from the expression, not by rewriting the comment text: gen_not_expr drops
  // a leading not2t, so the comment "!flag" gives "flag" and not "!(!flag)",
  // and from_expr re-parenthesises by precedence, so "x > 5" gives "!(x > 5)".
  static std::map<std::pair<std::string, std::string>, std::string>
    claim_negation;

  // Cross-run persisted covered-set (Item 2). covered_set holds the
  // (guard_str, location.as_string()) edge keys loaded from the JSON at
  // branch_coverage() entry; an edge present here is NOT re-instrumented
  // this run (its assert pair is skipped) but still counted in the
  // denominator via all_claims. covered_set_outpath, when non-empty, is
  // where bmc.cpp merge-writes the accumulated set at run end. Static so
  // the run-end report path (bmc.cpp) can read them like total_branch /
  // all_claims.
  static std::set<std::pair<std::string, std::string>> covered_set;
  static std::string covered_set_outpath;

  // ---- Complete-path coverage: CONTENT-ADDRESSED cross-run path keys ----
  //
  // Branch coverage keys an edge by (condition text, location) — content
  // addressed already. Complete-path coverage used to key a path by its ordinal
  // `enc`, which is a position in one particular enumeration. That is unsafe
  // across runs: the decision set has changed three times (short-circuit
  // operands, the ABI non-payable gate, internal-call expansion), and each
  // change RENUMBERS every path. A key witnessed under one numbering then
  // silently designates a DIFFERENT path under the next, so a path can be
  // skipped as "already covered" when nothing has covered it — no crash, no
  // warning, coverage simply reads low. An ordinal key cannot be made safe;
  // it has to be replaced.
  //
  // The stable key is the decision SEQUENCE itself:
  //     unit signature, then per decision (site, polarity, occurrence index)
  // where `site` is the decision's SOURCE LOCATION (plus its operand index, for
  // several folded short-circuit operands sharing one location) rather than any
  // serial number. Enumeration order, added decisions elsewhere, and a changed
  // unit set then leave a path's key untouched: an old key either matches
  // exactly or does not match at all — it can never designate a different path.
  //
  // `path_covered_ids` holds the ids loaded from the previous run;
  // `path_stable_id` maps this run's claim key -> its stable id so the run-end
  // write-back can record exactly the paths that were witnessed.
  static std::set<std::string> path_covered_ids;
  static std::map<std::pair<std::string, std::string>, std::string>
    path_stable_id;
  static std::string path_covered_outpath;

  // On-disk schema of the complete-path covered set. 3 adds `payloads` (see
  // path_covered_payload, declared below beside path_ce_t). Versions <= 2 carry
  // ids only and are REFUSED on load: reading one would mark paths covered
  // while permanently reporting them as payload-less.
  static constexpr int PATH_COVERED_SET_VERSION = 3;

  // Fail-closed guard for the file above. The stable key protects against
  // RE-NUMBERING; it cannot protect against a change that alters what a path IS
  // (different source, a decision kind added to the set, a different loop or
  // recursion bound). This fingerprint covers exactly those, and a mismatch
  // DISCARDS the cache and recomputes — deliberately with no migration path,
  // because migration logic is where this class of silent error hides.
  static std::string path_cov_fingerprint;

  // Serialise the complete-path covered set + its fingerprint + every
  // witnessed path's CE PAYLOAD (atomic publish, same .tmp-then-rename
  // discipline as write_covered_set_atomic). No-op when no
  // --coverage-covered-set was given.
  //
  // `when` labels the call site in the line this prints ("mid-solve after
  // claim 3 of 8" / "at run end"). It is not decoration: the whole claim being
  // made is that the payload is on disk BEFORE the run ends, and a line that
  // does not say when it was written cannot distinguish that from the old
  // behaviour.
  //
  // The counts printed are READ BACK OUT OF THE PUBLISHED FILE, not taken from
  // the in-memory maps that produced it. A census of what the writer believes
  // it wrote would have gone on printing correct-looking numbers throughout the
  // period in which nothing called this function at all.
  static void write_path_covered_set_atomic(const std::string &when = "");

  // Was this claim's path already witnessed in an EARLIER round?
  //
  // Complete-path coverage does NOT use `covered_set` — that is the branch
  // metric's (condition, location) pair set, and solidity_path_coverage()
  // deliberately leaves it empty because an ordinal-keyed pair cannot survive a
  // re-numbering. The cross-run identity here is the content-addressed stable
  // id, so the test has to go through path_stable_id -> path_covered_ids.
  //
  // Getting this wrong is not cosmetic: a path skipped because a previous round
  // already witnessed it has NO verdict this run, so reading `covered_set`
  // (always empty) reports it as U — a path with a counterexample in hand,
  // filed under "we could not decide". Measured: with --coverage-covered-set
  // the file was never even written, so the whole cross-run mechanism was dead
  // in both directions.
  static bool
  path_witnessed_earlier(const std::pair<std::string, std::string> &claim_key);

  // ---- U REASON TOKEN: why is this path not witnessed? ----
  //
  // U is a FIRST-CLASS DELIVERABLE, not an internal diagnostic. The claim being
  // made is "every uncovered path carries a named reason; there is no
  // unexplained remainder". A U cell that can also absorb an implementation
  // defect makes that claim false — and it has absorbed one three times already
  // (ordinal-key mismatch, the RETURN exit, `rawBalances`). Each time the broken
  // case was indistinguishable from an honest solver timeout, because "we do not
  // know" is a legitimate, expected outcome.
  //
  // So every U must be classifiable, and a U that is not is a HARD FAILURE
  // rather than a quiet residue. The tokens:
  //
  //   named-obstacle      the unit is disqualified (the model admits an
  //                       execution the chain does not have). Not an unknown at
  //                       all — a declared exclusion that no verdict can change.
  //   bounded-holds       the solver proved no witness AT THIS EXPLORATION.
  //                       Honest and precise: it is not "unreachable", it is
  //                       "no witness within the declared tx/unwind bound from
  //                       the post-constructor entry state".
  //   solver-unknown      the solver returned UNKNOWN/error, or an inductive
  //                       step could not prove it. The genuine "we do not know".
  //   not-solved-this-run the claim was instrumented but never reached the
  //                       solver. The entry-liveness audit catches the case
  //                       where a WHOLE unit is in this state; this token is
  //                       exactly the per-claim residue it cannot see.
  //   unit-not-entered    the harness never entered this path's unit, so no
  //                       classification of the path itself means anything.
  //
  // ORDER MATTERS, and `unit-not-entered` must sit SECOND — directly under
  // named-obstacle and ABOVE the two tokens that would otherwise swallow it:
  //
  //   * a unit that is entered but whose claims hold vacuously gets verdict 'P'
  //     and would be filed `bounded-holds` — "no witness within the bound",
  //     which is a statement about the path when the truth is that nothing ran;
  //   * a unit whose claims were never generated at all (measured on a real
  //     benchmark: 120166 instrumented paths, ZERO verification conditions) has
  //     no verdict and would be filed `not-solved-this-run`.
  //
  // Both are strictly less informative than the real reason, so putting them
  // first loses the one fact worth having. The ordering rule is the same one
  // that puts named-obstacle at the top: when the unit is disqualified, or was
  // never entered, every finer classification of its paths is meaningless.
  //
  // WHAT USED TO FIRE THIS, AND WHY IT NO LONGER DOES. The live producer was
  // --focus-function: the pass instrumented every unit while the dispatcher
  // entered one, so the excluded units' paths reached the report and were filed
  // here (measured before the ordering was fixed, they were reported
  // `not-solved-this-run` on a run whose own log line one line earlier said the
  // unit had not been entered). Since --focus-function narrows INSTRUMENTATION
  // (see `focus_function`), those units produce no claims at all, so on a
  // COMPLETE run `units_not_entered` can only ever name the focused unit — and
  // that is a hard failure in audit_entry_liveness, not a token.
  //
  // The slot is therefore reached today only on a PARTIAL run, where the audit
  // downgrades its abort to a warning and the un-entered units are the ones the
  // run stopped before reaching. It is kept, and kept SECOND, because the
  // ordering argument above is about what the token MEANS, not about how often
  // it occurs: a claim of a unit nothing entered must not be filed under a
  // verdict-derived token whatever put it there.
  //
  // This is an INVARIANT ASSERTION, not a detector: the live tokens partition
  // the possible states by construction (a claim's verdict is one of
  // {none, P, U, F}, and F means witnessed, hence not U). It therefore needs no
  // fault injection — its value is as a tripwire for a future change that
  // introduces a U with no reason. THE PARTITION ARGUMENT ONLY HOLDS WHILE THE
  // CLASSIFICATION HAS NO CATCH-ALL: the implementation's `default` arm returns
  // "" so the caller aborts. Mapping `default` to any token instead would make
  // the abort dead code while everything still looked fine — do not add one when
  // a fifth verdict value appears; add an explicit case for it.
  //
  // Returns the token, or "" when the path is not a U (which callers treat as
  // the hard failure above).
  static std::string
  path_u_reason_token(const std::pair<std::string, std::string> &claim_key);

  // The token names, in report order. Printed in full every time, zeros
  // included, so a category that stops occurring is visible rather than absent.
  static const std::vector<std::string> &path_u_reason_tokens();

  // ENTRY-LIVENESS AUDIT — hard-fails when a unit's results are vacuous.
  //
  // Measured on a real benchmark (St1inch): 120166 paths were instrumented and
  // symex generated ZERO verification conditions — the harness never called any
  // unit. Every path was then reported "U", which is indistinguishable from an
  // honest solver timeout, so a completely empty run looked like a merely hard
  // one. Nothing crashed and nothing warned; the defect was found by a human
  // reading a log line.
  //
  // Two levels of the same failure, deliberately sharing ONE channel so that
  // implementing half of it cannot leave the other half silent:
  //   * a unit whose claims never reached the solver was never entered;
  //   * zero verification conditions overall is the extreme case of that, where
  //     EVERY unit was never entered.
  //
  // The witness is positive: a claim that produced a solver verdict proves the
  // unit was executed. Absence of a verdict is not read as "undecided" — for a
  // whole unit it is read as "this run says nothing about this unit", which is
  // a tool failure, not a result. Claims deliberately skipped via the cross-run
  // covered set are excluded, since not instrumenting them is intentional.
  //
  // This is also what currently makes it safe to keep `I` disabled: if
  // unreachability were ever emitted, a never-entered unit would have every one
  // of its `assert(tr != enc)` hold vacuously and be reported as PROVEN
  // INFEASIBLE — the most damaging wrong answer this pass could give. The audit
  // is the precondition for ever enabling it.
  // `focus_function` is --focus-function's value, empty when unset. With it set
  // the dispatcher deliberately calls only that entry, so every OTHER unit is
  // legitimately never entered and must not be treated as a defect — measured:
  // the first real use of this audit aborted on exactly that. The premise
  // ("a unit with instrumented claims should be entered") only holds for the
  // focused unit, so the check is narrowed to where it holds rather than
  // weakened everywhere.
  //
  // SINCE --focus-function ALSO NARROWS INSTRUMENTATION (see `focus_function`),
  // the excluded units have no claims, so `all_claims` holds the focused unit's
  // paths and nothing else. The `dead_by_design` branch below is therefore
  // unreachable on a complete run, and the audit's HARD FAILURE now covers the
  // focused unit itself — a strengthening: a focused run whose one unit was
  // never entered used to be reported as a normal per-method result and is now
  // a defect. The narrowing is KEPT rather than deleted because it is the
  // property that makes the abort safe, and deleting it would leave nothing
  // stating why the abort may fire here at all.
  //
  // It uses focus_selects_unit(), the same matcher the instrumentation
  // narrowing uses, so the two cannot disagree about which unit is focused.
  static void audit_entry_liveness(const std::string &focus_function);

  // Units the audit found had claims instrumented but NONE decided, i.e. the
  // harness never entered them — mapped to WHY, because the two causes are
  // opposite in nature and must not be collapsed:
  //
  //   "excluded by --focus-function"  INTENDED and declared. The narrowing says
  //                                   this unit is not supposed to be entered
  //                                   in this run; its paths are meant to be
  //                                   witnessed by the run that focuses on them
  //                                   and unioned via the covered set. Normal
  //                                   output, labelled `unit-not-entered`.
  //   "harness never entered it"      A DEFECT. Measured on a real benchmark:
  //                                   120166 instrumented paths, zero
  //                                   verification conditions. Hard failure.
  //
  // Keeping the reason (rather than a bare set) is what makes this usable BEYOND
  // token selection. The planned entry-liveness WITNESS — an `assert(false)` at
  // each unit's body head that must be refuted — has the rule "not refuted =>
  // hard failure", and applying that rule uniformly would abort the moment
  // anyone passes --focus-function. It must consult this same distinction:
  // SKIP the check for focus-excluded units, REQUIRE refutation for the rest.
  // Recording only "not entered" would force that layer to rediscover the split.
  //
  // Filled by audit_entry_liveness (which runs before any figure is printed).
  // Today the defect entries never survive to a reader — the audit aborts on
  // them — but both are recorded so the shape is right when that changes.
  static std::map<std::string, std::string> units_not_entered;

  // ---- Solidity complete-path coverage: tri-state (F/I/U) reporting ----
  //
  // `reached_claims` alone cannot distinguish "proven unreachable" from
  // "could not decide": both are simply absent from it. The tri-state
  // report needs the per-claim solver verdict, so multi_property_check
  // records it here as it solves (keyed by claim_sig == "msg\tloc"):
  //   'F' — refuted (P_SATISFIABLE): the path is feasible and a
  //         counterexample (concrete input) is in hand.
  //   'P' — proven (P_UNSATISFIABLE) AT THE CURRENT EXPLORATION. This is a
  //         CANDIDATE for I and nothing more: it means "no witness within this
  //         tx/unwind bound, from this entry state", NOT unreachability.
  //         NOTHING currently upgrades it. In particular --solidity-max-tx 0
  //         does NOT: coverage rewrites the dispatcher back-edge to a SKIP, so
  //         that flag explores ONE transaction — fewer than
  //         --solidity-max-tx 2. Every 'P' is therefore reported as U with
  //         `bounded_holds: true`. See path_cov_can_prove_unreachable() in
  //         bmc.cpp, which is the single place to change if a havoc'd-entry or
  //         loop-live exploration mode is ever added.
  //   'U' — undecided: solver UNKNOWN/error, or the inductive step could
  //         not prove it.
  // Written under a mutex (--parallel-solving runs jobs on threads).
  static std::map<std::string, char> claim_outcome;
  static std::mutex claim_outcome_mutex;

  // Complete paths whose exit is a detected custom-error revert
  // (`revert E()` reaching a `#sol_error` call). Keyed like all_claims, so
  // the report can label each path's exit_kind normal|revert. Filled by
  // solidity_path_coverage() at instrumentation time.
  static std::set<std::pair<std::string, std::string>> revert_paths;

  // Complete paths that exit through a ROLLBACK revert: `require(cond)` /
  // `require(cond,"msg")` / `revert("msg")` lower to a block that restores the
  // entry snapshot (`*this = _sol_save_this`) and jumps to END_FUNCTION. These
  // reach END_FUNCTION like a normal exit, so without this set they would be
  // reported `exit_kind: "normal"` — actively wrong, since the transaction
  // reverts. Unlike `revert_paths` (custom errors) the rollback IS modelled,
  // so their post-state is the correctly restored one and needs no warning.
  static std::set<std::pair<std::string, std::string>> rollback_revert_paths;

  // Paths whose exit shape cannot be classified: they reach END_FUNCTION while
  // skipping the epilogue and carry no rollback restore. A `require` failing
  // before any state write and a plain early `return` both compile to exactly
  // this, with nothing on the edge to separate them. Reported as
  // `exit_kind: "undetermined"` — labelling them "normal" would assert that a
  // reverted transaction succeeded.
  static std::set<std::pair<std::string, std::string>> undetermined_exit_paths;

  // Complete paths POSITIVELY confirmed to exit normally. The three sets above
  // say what went wrong; this one says nothing did, and it exists so that no
  // consumer has to infer normality from absence.
  //
  // That inference is not merely inelegant, it is wrong across coverage modes:
  // a branch-coverage claim appears in none of these sets either, so "in
  // all_claims and in no failure set" calls every branch claim normal. Measured
  // — it turned three branch-coverage regressions red on the first attempt.
  //
  // Only complete-path coverage fills this. Its consumer is the Foundry
  // emitter, which drops the revert-tolerant try/catch and emits the call bare
  // when the exit is confirmed normal, so a revert at run time fails the test.
  // That makes this set the single fact authorising a generated test to assert
  // anything at all, which is why it is recorded rather than derived.
  static std::set<std::pair<std::string, std::string>> normal_exit_paths;

  // Paths along which the MODEL AND THE EVM DISAGREE, keyed like all_claims,
  // with the value naming the obstacle.
  //
  // These are not imprecision — they are paths where a counterexample would
  // describe an execution that does not exist on-chain, so a test built from
  // one is red when run while being labelled certified. That is the single
  // failure mode this pipeline must never produce, and it is why the marking is
  // not merely reported: a marked path must be excluded from the sibling set
  // used for the stage-3 subtraction AND must not be turned into a test.
  // Marking without excluding would be worthless.
  //
  // Counted and reported as an ABSOLUTE number, never folded into a coverage
  // ratio: an obstacle is not partial credit.
  //
  // Marking is PER PATH, not per unit — a path of the same unit that never
  // walks the offending site is unaffected and stays fully usable. The rule is
  // conservative: any path whose decision sequence passes through the site is
  // marked.
  static std::map<std::pair<std::string, std::string>, std::string>
    named_obstacle_paths;

  // Paths whose certified region is NARROWER than it would otherwise be,
  // because their unit lost paths to the goal cap. Keyed like all_claims.
  //
  // This is deliberately NOT named_obstacle_paths, and the reason is mechanical.
  // Certification is a query — `assume(L <= x <= U); assert(tr == pi)` — and the
  // goal cap limits only how many EXIT ASSERTS are emitted. Phase-1 accounting
  // still updates `tr`/`cnt` on every decision of every path, dropped ones
  // included, so an input that walks a dropped path carries that path's number
  // at the exit, the query fails on it, and the candidate interval is rejected
  // and shrunk. The query never needed the dropped path to have been enumerated.
  //
  // A `require` lowered to a control-flow-free assume is the opposite case and
  // stays an obstacle: there the reverting execution does not exist in the model
  // at all, so no query can see it and no interval can be shrunk away from it.
  // Existing-but-unenumerated and non-existent are different failures, and only
  // the second can ship a test that is red on the unmodified contract.
  static std::map<std::pair<std::string, std::string>, std::string>
    truncation_weakened;

  // ---- DEGRADATION: which call points a unit gave up to fit its budget ----
  //
  // Degradation and TRUNCATION are two different mechanisms with two different
  // soundness stories, so they get two separate reports and are never merged:
  //
  //   * DEGRADATION withdraws call points BEFORE enumeration. The callee stays
  //     a call, so symex still executes it; it simply stops contributing
  //     decisions to the caller's path identity. The path classes get COARSER
  //     but still partition the input space (two different decision sequences
  //     still differ in polarity at their first differing index; execution is
  //     still deterministic), so the enumeration stays sound. What is lost is
  //     assertion STRENGTH, and it is lost at named, reported places rather
  //     than everywhere.
  //   * TRUNCATION drops enumerated paths at the goal cap. Those paths exist
  //     and symex will execute them; they are simply missing from the sibling
  //     set.
  //
  // The order is fixed: degradation fires FIRST, truncation is the last-resort
  // backstop. In the intended steady state truncation never fires at all — so
  // if it does fire, that is a result in its own right (the degradation policy
  // was not aggressive enough for that unit) and is reported as such, never
  // folded into the degradation report.
  //
  // Keyed by unit id; the value names each withdrawn call point (callee plus
  // the source location of the call), because "this unit was degraded" is not
  // actionable while "this unit stopped recording the decisions of THIS call"
  // is.
  static std::map<std::string, std::vector<std::string>> degraded_call_sites;

  // The CE payload of a witnessed (F) path: the concrete values that make the
  // path execute, harvested from the solver model while it is still live.
  // This is the half of the report a downstream generaliser actually consumes
  // — a path id alone says "reachable", these values say "with WHAT".
  //   inputs      — first nondeterministically-sourced value per symbol
  //                 (the harness's chosen call arguments / environment)
  //   final_state — LAST value written to each contract state variable
  //                 (`this->x`) on this path, i.e. the post-state
  // Both are recorded ONLY from concrete model values; a symbolic or missing
  // value is dropped rather than guessed.
  //
  // `sliced`/`compact_trace`/`payload_symbols_protected` record HOW the trace
  // was produced, because all three affect what can be harvested: the symex
  // slicer keeps only steps the CLAIM depends on, and a path claim's guard
  // mentions only the ghost accumulators, so state-variable writes would be
  // sliced away — unless the payload symbols were exempted from slicing
  // (protect_ce_symbols, which --cov-report-json turns on) or slicing was
  // switched off entirely. Without these flags an empty final_state would be
  // ambiguous between "this path writes no state" and "the writes were sliced
  // away".
  struct path_ce_t
  {
    // Solidity call arguments only: a nondet-sourced write to a symbol whose
    // mangled id has the `sol:@C@<C>@F@<f>@<name>` shape. Harness plumbing
    // (allocator tables, dispatcher choice bits, temporaries) is NOT an input
    // and is counted in `dropped_internal` instead of polluting this list.
    std::vector<std::pair<std::string, std::string>> inputs;
    // EVM environment the path was witnessed under (msg.*/tx.*/block.*),
    // kept separate because it is context, not an argument.
    std::vector<std::pair<std::string, std::string>> env;
    // How many nondet values were classified as harness-internal and dropped.
    // Reported so the omission is visible rather than silent.
    size_t dropped_internal = 0;
    // What the external calls on this path RETURNED. Under --unbound the
    // frontend models an external call as a nondet return plus a nondet
    // re-entry, so this value is chosen by the model, not by the caller —
    // keeping it out of `inputs` is the difference between a replayable test
    // and one that tries to pass an unpassable argument.
    std::vector<std::pair<std::string, std::string>> extcall_returns;
    // Contract state at the moment this path's function was ENTERED. A path
    // guarded by state that an earlier transaction established is only
    // reproducible if the entry state is known too.
    std::vector<std::pair<std::string, std::string>> entry_storage;
    // False when the entry marker was not seen in the trace (e.g. the entry
    // assignment was sliced, or the path was witnessed in an earlier round), so
    // an empty entry_storage is never read as "the contract started empty".
    bool entry_storage_known = false;
    std::vector<std::pair<std::string, std::string>> final_state;
    // State variables this path provably WROTE but whose value could not be
    // rendered as a constant (mappings / dynamic arrays lower to infinite-array
    // globals whose model value is the whole store). Listed by name so the
    // reader knows the variable changed: omitting them entirely would let a
    // consumer infer "unchanged", which is a silent wrong conclusion.
    std::vector<std::string> state_written_unrendered;
    bool sliced = true;
    bool compact_trace = true;
    // The symbols this payload is built from (contract objects, contract-scope
    // mapping/array stores, msg./tx./block.) were registered as no-slice names,
    // so `sliced == true` here does NOT mean the payload was cut down. This is
    // what makes an empty `final_state` readable as "this path writes no state"
    // rather than "the writes were sliced away".
    bool payload_symbols_protected = false;
    // The harvest stopped at this path's own violated assert, so values from
    // any LATER transaction in a multi-tx harness cannot leak in. False means
    // the assert was not found in the trace and the whole trace was scanned —
    // the post-state may then belong to a later transaction.
    bool scoped_to_claim = false;
    // This path exits through a custom-error `revert E()`, which the Solidity
    // frontend lowers to a `#sol_error` callee containing only ASSUME(false)
    // — with NO state rollback (unlike require/revert("msg"), which restore
    // `*this` from an entry snapshot). So the harvested post-state is the
    // state AT THE REVERT POINT, not the EVM post-state (on-chain every write
    // in the reverted transaction is undone). Reported, never silently used.
    bool revert_pre_rollback = false;
    // ---- THE UNIT'S OWN RETURN VALUE ON THIS PATH ----
    //
    // MEASURED before this field existed, on notes/coverage/poc/
    // P19_ReturnShapes.sol unit `tern_lit` with --verbosity coverage:9: the
    // harvest classified 208 assignments and NOT ONE of them was the unit's
    // return. Every `return_value$` symbol it did see belongs to the C library
    // (_nondet_uint, _sol_per_tx_reseed, _ESBMC_get_unique_address).
    //
    // The cause is not that the value is harvested and dropped for want of a
    // bucket -- that is the extcall_returns story and it does not apply here.
    // The value NEVER EXISTS as an assignment: the dispatcher calls a unit with
    // no lvalue (`FUNCTION_CALL: tern_lit(&obj, NONDET, NONDET)`) and the RETURN
    // instruction carries an EXPRESSION, not a write to any symbol. The rival
    // explanation, "it is written but after the harvest's break", is REFUTED by
    // the same goto dump.
    //
    // So the instrumenter materialises it into a ghost at the RETURN site,
    // placed BEFORE the path asserts precisely so it falls inside the harvest's
    // window (the walk stops at this path's own assert).
    //
    // `return_value_known` is a SEPARATE flag and it is mandatory. A `void`
    // unit, an aggregate return the instrumenter does not render, and a path
    // that reverts without reaching a RETURN all leave the string empty, and an
    // empty string alone would read as "this unit returns nothing" -- a claim
    // about the contract made out of an absence in the harvest.
    std::string return_value;
    bool return_value_known = false;
  };
  static std::map<std::string, path_ce_t> path_ce;

  // ---- THE COUNTEREXAMPLE PAYLOAD, PERSISTED WITH THE ID ----
  //
  // The cross-run file used to hold ONLY stable ids. That is enough to SKIP a
  // path, and not enough to keep the thing the skip is protecting: a path
  // recorded as covered but carrying no inputs can never produce a test, and
  // the report says so with `payload_absent_reason` -- permanently, because the
  // round that could still have produced the payload is the one that just
  // skipped the path.
  //
  // So enabling the mid-solve write WITHOUT this map converts a witness lost to
  // an OOM into a payload-less `F` that no later round will ever repair -- a
  // regression dressed as a fix. The payload is therefore written by the SAME
  // atomic publish as the id, keyed by the same stable id, and the file's
  // `version` is bumped so a payload-free file from an older build is REJECTED
  // rather than silently read as "these paths have no inputs".
  //
  // Loaded from the file at instrumentation time; written back as the UNION of
  // loaded and newly witnessed on every publish, so a round that does not
  // re-instrument a path does not drop that path's payload either.
  // EVERY witness of an F claim, in enumeration order, with `path_ce`'s entry
  // as element 0. Non-empty only for a claim that produced a payload; under
  // `--all-witnesses` it holds up to `--max-witnesses` of them.
  //
  // A SECOND MAP RATHER THAN A CHANGED TYPE ON `path_ce`. Four consumers read
  // `path_ce` -- the Foundry emitter, audit_certify_witness, the CE journal and
  // the covered-set writer -- and three of them want exactly one witness.
  // Widening the existing type would have been a change to all four at once,
  // and the one that wanted the others is the report.
  static std::map<std::string, std::vector<path_ce_t>> path_ce_all;

  static std::map<std::string, path_ce_t> path_covered_payload;

  // The payload an EARLIER round persisted for this claim's path, or nullptr.
  // Goes through path_stable_id -> path_covered_payload, i.e. exactly the
  // indirection path_witnessed_earlier uses, so the two can never disagree
  // about which path a claim is.
  static const path_ce_t *
  path_payload_earlier(const std::pair<std::string, std::string> &claim_key);

  // ---- THE COUNTEREXAMPLE JOURNAL (`cov-ce-journal.json`) ----
  //
  // `cov-report.json` is written exactly once, from report_coverage, which sits
  // AFTER the per-claim job loop and INSIDE the try that an OOM unwinds. A run
  // that dies therefore keeps nothing -- measured: a whole-contract run died
  // 51.5% through the solve having REFUTED 5 of that contract's 15 paths, and
  // discarded all five.
  //
  // The cross-run covered set above would have kept them, but only for a caller
  // that passed --coverage-covered-set, and no collector in this project ever
  // has. The journal has no such gate: it is written whenever the run asked for
  // the counterexample payload at all (--cov-report-json), refreshed by an
  // atomic .tmp+rename at the moment each path is WITNESSED, and never read
  // back in, so it cannot accumulate across runs or change what a re-run does.
  //
  // It is explicitly INCOMPLETE until the run says otherwise: `complete` is
  // false on every incremental write and true only on the one written beside
  // the final report. A journal read as a finished report would understate
  // every count in it.
  //
  // Cost is one serialisation of the witnessed set per witness, i.e. quadratic
  // in |F|. |F| is single digits per unit on every contract measured so far;
  // the same shape is already accepted for write_path_covered_set_atomic.
  static std::string path_ce_journal_path;
  static void
  write_path_ce_journal_atomic(const std::string &when, bool complete);

  // How many claims this run has HANDED TO THE SOLVER AND GOT AN ANSWER FOR,
  // and how many it had. Atomics, because the two readers that need them most
  // are a signal handler (which may not take a lock or touch a std::map) and
  // the journal writer (which runs on whichever job thread witnessed a path).
  //
  // `live_decided` is the numerator of "how much of this run's work would be
  // thrown away if it died right now" — the number that turned out to be 938
  // on the run this whole change is about.
  static std::atomic<size_t> live_decided;
  static std::atomic<size_t> claims_total_atomic;

  // ---- THIS RUN DID NOT CONCLUDE, AND EVERYTHING BELOW IS CONDITIONED ON IT --
  //
  // Empty means the solve loop ran to the end. Non-empty names how it died, and
  // is the single fact that changes three otherwise-unrelated behaviours:
  //
  //   * `cov-report.json` is stamped PARTIAL (report["partial"], and the same
  //     under `summary`). A partial report read as a complete one would deflate
  //     every numerator computed from it, silently, and it lands under the same
  //     filename a complete report does — so the marker is the only thing
  //     separating them and it is written in both directions, never omitted.
  //   * a claim with NO verdict is filed `run-died-before-solving` rather than
  //     `not-solved-this-run`. Today that token means "simplified away at symex
  //     time", which is a fact about the CLAIM; a claim the run never got to is
  //     a fact about the RUN, and collapsing the two makes the report unable to
  //     explain the very thing it is reporting.
  //   * audit_entry_liveness stops aborting. Its premise -- "a unit with
  //     instrumented claims should have been entered" -- only holds for a run
  //     that reached the end of its job loop. On a run that died at claim 1 of
  //     N almost every unit is legitimately un-entered, and aborting there would
  //     destroy the partial report on its way out.
  static std::string path_cov_partial_reason;

  // The claim comments the per-claim solve loop was actually given, i.e. the
  // asserts that survived simplification and reached the equation. Filled once,
  // before the first solve, by walking the equation.
  //
  // WITHOUT IT THE PARTIAL RUN'S U-REASONS ARE WRONG, and wrong in the
  // direction that overstates the damage. MEASURED on aqua at 8 g: 2846 paths,
  // of which 1024 never became a VCC at all (the simplifier folded them away at
  // symex time) and 1822 reached the loop. Classifying every undecided claim as
  // `run-died-before-solving` reported 1826 paths as lost to the death when
  // ~901 were -- and the other 925 are lost to something no budget can fix.
  // Two different facts, two different next actions, one cell.
  //
  // RESIDUAL IMPRECISION, stated rather than left to be found: a claim can be
  // folded away on one symex branch and generated on another. Such a claim IS
  // in this set, so if the run dies before reaching it, it is filed
  // `run-died-before-solving` -- which is right. The reverse (a claim in
  // neither population) cannot occur: a claim absent from the equation was
  // never queued.
  static std::set<std::string> claims_in_solve_loop;

  // ---- THE PER-CLAIM SOLVER BUDGET ----
  //
  // Seconds, 0 = unlimited (--path-cov-claim-timeout, default 120). Published
  // in the report's `summary.bound` because a capped run's U counts are NOT
  // comparable with an uncapped run's: some of its U's are "we stopped asking",
  // and a reader comparing two reports without this number would treat that as
  // "no witness exists". `claim_budget_exceeded` is how many claims were
  // abandoned, so "the cap was on" and "the cap fired N times" are separate
  // statements -- a cap that never fires costs nothing and changes no verdict.
  static size_t claim_budget_seconds;
  static std::atomic<size_t> claim_budget_exceeded;
  // Which enforcement each backend got, in the tool's own words, so a reader
  // never has to infer it: a native solver limit and a watchdog interrupt have
  // different granularity and different failure modes, and a backend with
  // NEITHER must say so rather than silently run unbounded.
  static std::string claim_budget_mechanism;

  // ---- --path-cov-arith-resolve: THE CHAIN REJECTS THIS COUNTEREXAMPLE ----
  //
  // A witnessed path whose model violates an enabled arithmetic check describes
  // an execution the EVM does not have: the chain reverts with Panic 0x11
  // (overflow) or 0x12 (division by zero) where the model wraps or returns
  // bvudiv's total-function value. The emitted Foundry case then asserts a
  // NORMAL exit for a transaction that reverts, and is RED on the unmodified
  // contract -- measured, three times across the PoC set.
  //
  // `arith_revert_only_paths` holds the paths for which the re-solve came back
  // UNSAT, i.e. the ones PROVEN reachable only by overflowing. That is a
  // DECIDED property of the path and it gets its own cell: folding it into U
  // would file a proof under "we could not decide", which is the exact failure
  // the U-reason tokens exist to prevent. Keyed like all_claims.
  //
  // The three counters are the COST, and they are printed rather than inferred.
  // Nobody knew, when this was designed, whether the re-solve would fire on
  // three claims or three thousand; a mechanism whose price is unmeasured is
  // one nobody can decide to keep.
  static std::set<std::pair<std::string, std::string>> arith_revert_only_paths;
  static std::atomic<size_t> arith_resolve_queries;   // re-solves attempted
  static std::atomic<size_t> arith_resolve_replaced;  // better witness found
  static std::atomic<size_t> arith_resolve_ms;        // wall time, ms
  // How many arithmetic-check claims the equation carried at all. Printed even
  // when it is ZERO, and especially then: zero means no check was enabled or
  // none reached this unit, and without the number a run that re-solved nothing
  // is indistinguishable from a run that had nothing to re-solve.
  static std::atomic<size_t> arith_conditions_seen;
  // How many counterexamples were REFUSED to the Foundry emitter because their
  // path is arith-revert-only. Counted, never silent: goto_coverage.h's own
  // rule for the obstacle machinery is that "a marked path must not be turned
  // into a test -- marking without excluding would be worthless", and a
  // suppression that leaves no trace is indistinguishable from a path that was
  // never witnessed.
  static std::atomic<size_t> arith_revert_only_suppressed;

  // How many times a DECIDED verdict was kept because a later solve of the same
  // claim key came back without one. Printed on every path-coverage run,
  // including at zero: "the guard fired N times" and "the guard was never
  // needed" are different statements, and a guard whose only evidence is that it
  // compiles is the shape this pass has shipped twice.
  //
  // Non-zero means the same claim key reached the solve loop more than once --
  // which is itself a defect (duplicate instrumentation), measured on st1inch as
  // 10 VCCs for 5 paths. So this counter is simultaneously the fix's effect and
  // the other defect's detector.
  static std::atomic<size_t> verdicts_preserved;

  // Signal-safe snapshot for path coverage, mirroring branch coverage's
  // (branch_cov_active / total_branch_atomic / live_reached). Written at the
  // end of instrumentation and in the per-claim job; read ONLY by the signal
  // handler, which may take no lock and may not walk a std::map.
  static std::atomic<bool> path_cov_active;
  static std::atomic<size_t> total_paths_atomic;
  static std::atomic<size_t> live_F;

  // Each enumerated path's DECISION DEPTH, keyed like all_claims.
  //
  // Reported because the next stage cannot be driven without it: every stage-2
  // query identifies a path by `tr == enc && cnt == depth`, and `enc` alone does
  // not identify it — the `cnt` conjunct is what stops a longer path whose
  // 64-bit `tr` wrapped from firing a shorter path's claim. A driver reading
  // only the report had to be told the depth by hand, which is exactly the kind
  // of out-of-band knowledge that makes an interface not one.
  static std::map<std::pair<std::string, std::string>, uint64_t>
    path_decision_depth;

  // ---- THE ORDERED DECISION SEQUENCE BEHIND A PATH ----
  //
  // `path_id` is `enc`, and `enc` is a pure bit accumulator: the k-th decision's
  // ARM is `(enc >> (depth-k)) & 1` arithmetically, but nothing about the source
  // location is mixed into it. Which SITE each bit came from is path-dependent
  // (bit 3 of one path and bit 3 of a sibling come from different instructions
  // once their prefixes diverge), so no external consumer can recover it from
  // the report. The information exists only here, during enumeration.
  //
  // It is needed for the one comparison this pass has never been able to make:
  // projecting the witnessed (F) path set onto the DECISIONS those paths walk,
  // which is the only common denominator with --branch-coverage. Without it the
  // two metrics cannot be put on one scale at all.
  //
  // THE STORAGE IS INTERNED, and that is what makes it affordable. A unit can
  // enumerate 120166 paths; storing a location string per path per decision is
  // the reason the pre-existing (log-only) recorder is gated off by default.
  // Here the DESCRIPTORS live in a per-unit table whose size is the number of
  // distinct decision sites (tens), and the per-prefix map holds a 32-bit index.
  //
  // Keyed by the PREFIX `enc` value the decision produced, so a path's sequence
  // is read off by walking `enc >> (depth-1)`, `enc >> (depth-2)`, ..., `enc`.
  // Two paths sharing a prefix share those keys by construction, so one entry
  // serves every path through it and the map cannot be ambiguous. Polarity is
  // not stored: it is the key's own low bit.
  struct path_decisiont
  {
    // `location.as_string()`, i.e. exactly the string bmc.cpp's
    // parse_claim_location() already splits into file/line/column/function.
    // Deliberately not pre-split here: a second parser is a second thing that
    // can disagree with the first.
    std::string loc;
    // Each arm's claim text, built with the same from_expr/gen_not_expr the
    // branch metric uses.
    //
    // Both arms are published because the mapping is INVERTED and inverting it
    // in the consumer is a silent, plausible-looking error: a probe assert(P)
    // fails when P is false, so `assert(guard)` covers the FALL-THROUGH edge and
    // `assert(!guard)` the GOTO-TAKEN edge. Path polarity TRUE (taken) therefore
    // corresponds to the claim keyed on the NEGATED guard. Getting it backwards
    // still produces a number.
    //
    // ⚠ IT IS DIAGNOSTIC, NOT A JOIN KEY AGAINST --branch-coverage. MEASURED on
    // regression/esbmc-solidity/solidity_path_cov_exit_kinds, same contract,
    // same source line, the two modes report DIFFERENT text for the SAME
    // decision:
    //
    //     branch coverage       "!(a != 0)"   /  "a != 0"
    //     path coverage       "!(!(a != 0))"  /  "!(a != 0)"
    //
    // The guards are one `not` apart because --solidity-path-coverage turns on
    // the revert-observation gate, which lowers `require` to a different goto
    // shape. A plain `if` agrees exactly (verified on the same run: `v > 1` and
    // `!(v > 1)` match both sides verbatim), so a text join would work on part
    // of the corpus and quietly drop every `require` decision on the rest —
    // producing a lower number with no error.
    //
    // A projection onto branch coverage must therefore join on LOCATION, which
    // is also what the comparison's own metric is defined in terms of (unique
    // source lines reached, capped at the file's decision count).
    std::string cond_arm_true;  // polarity 1 (taken)  -> this branch claim
    std::string cond_arm_false; // polarity 0 (fall-through)
    // Operand index within the site, for several folded short-circuit operands
    // sharing one location.
    unsigned sub = 0;
    // The synthesised ABI non-payable gate (`msg_value == 0`). It is a decision
    // of the PATH metric with NO branch-coverage counterpart, and its location
    // is COPIED from the unit's first body instruction — so a consumer matching
    // on location alone would credit itself with a real decision sitting on that
    // line. Flagged at the source rather than left for the consumer to guess
    // from the condition text.
    bool synthetic_abi_gate = false;
  };
  // unit id -> that unit's interned decision descriptors.
  static std::map<std::string, std::vector<path_decisiont>> path_decision_table;
  // unit id -> (prefix enc -> index into the table above).
  static std::map<std::string, std::map<uint64_t, uint32_t>> path_decision_index;

  // ---- THE PER-PATH EMIT SEQUENCE (R0's event rung) ----
  //
  // Same two-level shape as the decisions above and for the same reason: the
  // names are interned (a contract has tens of distinct events) and only the
  // per-prefix keys are stored, so nothing per-path is copied into the DFS
  // stack. That copy is what path_decision_table exists to avoid, measured on
  // units of 2733 and 120166 paths.
  //
  // The INNER key is the emitting instruction's PROGRAM POSITION, not a list
  // position, and that buys the two properties a vector cannot give:
  //
  //   IDEMPOTENCE  a prefix is re-walked once per branch explored beneath it,
  //                so an appending scheme multiplies every event by the number
  //                of paths under it. Assigning to a position overwrites the
  //                same slot with the same value.
  //   ORDER        several emits between two decisions come back in program
  //                order rather than as an unordered set.
  //
  // ⚠ IT IS NOT A COMPLETE RECORD OF WHAT THE PATH EMITS. The qualified
  // spelling `emit L.E(x)` becomes a code_skipt() in the front end
  // (solidity_convert_expr.cpp) and reaches the goto program carrying nothing,
  // so it is INVISIBLE here. A consumer must therefore never read an empty
  // array as "this path emits no events" — only as "no unqualified emit was
  // recorded on it".
  //
  // unit id -> interned event names.
  static std::map<std::string, std::vector<std::string>> path_event_table;
  // unit id -> (prefix enc -> (program position -> index into the table)).
  static std::map<std::string, std::map<uint64_t, std::map<uint32_t, uint32_t>>>
    path_event_index;

  // Item 2e: serialize covered_set to covered_set_outpath crash-safely
  // (write a .tmp then atomic rename). Called both incrementally as
  // each edge is witnessed P_SATISFIABLE (bmc.cpp) and once at run end,
  // so a mid-run kill still persists every edge proven so far and
  // bounded re-runs accumulate monotonically. No-op if outpath empty.
  static void write_covered_set_atomic();

  // "data even on UNKNOWN": an external kill (SIGALRM from esbmc's own
  // --timeout, or SIGTERM/SIGINT from timeout(1)/CI/orchestrator) lands
  // mid-solve, so report_coverage (the stdout "Branch Coverage:" line,
  // only reached at a normal conclude/exhaustion point) never runs even
  // though the numerator is already known. The signal handler reads
  // this async-signal-safe snapshot and emits a SOUND LOWER BOUND
  // before _exit. It is EXACT once any report_coverage has run (every
  // such call re-syncs the active counter to its authoritative value);
  // between the first per-claim hook and the first report it is a
  // lower bound. Handler does only atomic loads + write(2); it never
  // touches the std::set (mid-mutation under SIGALRM).
  //   branch_cov_active   — set once branch-coverage mode is live
  //   total_branch_atomic — |all_claims| (denominator), set at
  //                          instrumentation, pre-solve; mirrors
  //                          total_branch
  //   covered_set_mode    — true iff --coverage-covered-set given; it
  //                          selects which numerator the handler reads
  //   live_reached        — DEFAULT mode numerator == reached_claims
  //                          .size() (the canonical bmc.cpp:901 count,
  //                          which intentionally includes non-universe
  //                          entries); updated under reached_claims_
  //                          mutex at the per-claim hook
  //   covered_run         — COVERED-SET mode numerator == count of
  //                          universe edges newly witnessed+persisted
  //                          THIS run (covered_set.emplace().second at
  //                          the Item 2e hook). A loaded prior set only
  //                          raises true coverage, so this is a sound
  //                          lower bound on the covered-set authoritative
  //                          |all_claims ∩ (covered_set ∪ reached)|.
  static std::atomic<bool> branch_cov_active;
  static std::atomic<size_t> total_branch_atomic;
  static std::atomic<bool> covered_set_mode;
  static std::atomic<size_t> live_reached;
  static std::atomic<size_t> covered_run;

  std::string target_function = "";
  bool cov_assume_asserts = false;

  // Context for creating+registering ghost snapshot symbols in
  // solidity_path_coverage(). namespacet only exposes a const context, so
  // the pass needs a mutable contextt& to move() new symbols in (pattern:
  // assign_params_as_non_det / symbol_generator). Set by the dispatch site
  // (esbmc_parseoptions.cpp) where `context` is available; nullptr => the
  // pass aborts with an actionable error rather than silently no-op.
  contextt *cov_context = nullptr;
  // Per-unit budget on enumerated complete paths (--path-cov-max-goals).
  // TWO mechanisms are keyed off it, in a fixed order (see
  // degraded_call_sites): degradation withdraws call points until the unit
  // fits, and only if that fails does the enumeration truncate at the cap. On
  // truncation the dropped count is always reported — never a silent cut.
  size_t path_cov_max_goals = 10000;
  // Loop bound for path enumeration: each back-edge is followed at most this
  // many times per path, so complete paths are enumerated up to this many
  // loop iterations. Set from --unwind by the dispatch (default 4). Must
  // match the symex unwind bound, or enumerated paths and solver-explored
  // paths disagree.
  size_t path_cov_unwind = 4;
  // When non-empty (set from --contract), branch_coverage() instruments
  // ONLY decisions whose lexically-declaring Solidity contract equals this
  // name (location's "sol_decl_contract", stamped by the frontend). This
  // makes per-contract branch coverage count C's own source decisions
  // only; inherited/library decisions are attributed to their own
  // declaring contract and excluded. Empty => no scoping (unchanged
  // whole-unit behaviour, e.g. C/C++/no --contract).
  std::string scope_contract = "";

  // ---- --focus-function NARROWS WHAT IS INSTRUMENTED, not only what is
  // ---- ENTERED ----
  //
  // Set from --focus-function; empty when unset. The frontend's dispatcher
  // filter (solidity_convert_constructor.cpp) already narrows which entry the
  // harness may call. This narrows the OTHER half: solidity_path_coverage()
  // enumerates and instruments path claims for THIS unit only.
  //
  // WHY, and it is not performance. A focused run used to instrument the whole
  // contract's path set, so the numbers it published were CONTRACT-level while
  // reading as the unit's. MEASURED on aqua `--focus-function dock`:
  //
  //     Complete Paths : 2846    Reached : 2    Path Coverage: 0.07%
  //
  // 2783 of those 2846 belong to units the dispatcher cannot enter in this run,
  // so no exploration could ever witness them. The honest denominator is
  // `dock`'s own 63 paths, i.e. 3.17% -- the published metric was wrong by 45x,
  // and `summary.paths_total` carried the same contract-level number into
  // cov-report.json, where it has already been misread as the unit's.
  //
  // The time saving is a SIDE EFFECT AND A SMALL ONE; do not quote it as the
  // reason. Focus already narrowed the solve stage (symex never reaches a
  // non-focused unit's assert -- aqua `dock`: 2846 instrumented, 63 VCCs), so
  // the only phase this can compress is the instrumentation pass itself, billed
  // to `GOTO program processing time`: measured 0.20-1.80 s across the corpus,
  // against a `GOTO program creation time` of 1.1-13.4 s that is the FRONTEND's
  // Solidity->GOTO conversion and is not affected by this at all.
  //
  // ---- WHAT THIS MUST NOT NARROW: the EXPANSION loop ----
  //
  // solidity_path_coverage() walks goto_functions TWICE: first to expand
  // internal calls physically into each unit (sol_path_inlinet::expand_here)
  // and to run degradation, then to enumerate and instrument. ONLY THE SECOND
  // LOOP IS NARROWED.
  //
  // The first must not be, and the reason is mechanical rather than cautious:
  // `expand_here` copies the callee's body AS IT IS AT THAT MOMENT, and the
  // expansion loop rewrites each unit's body in place as it goes. So a unit C
  // that the loop reaches BEFORE a unit F is already expanded when F splices it
  // in, and F gets one further level of call depth for free. Skipping
  // non-focused units there -- or merely skipping their degradation, which also
  // rewrites their bodies -- would change what lands inside the focused unit,
  // and every `enc` would silently mean something else: a smaller, faster run
  // with different answers. Leaving the loop alone keeps the focused unit's body
  // BIT-IDENTICAL to its body in a whole-contract run, which is also what makes
  // the covered-set argument below hold.
  //
  // The price is stated rather than optimised away: a focused run still pays the
  // whole contract's expansion and degradation cost.
  std::string focus_function = "";

  // ---- --path-cov-instrument-only: DISPATCH WIDE, INSTRUMENT NARROW ----
  //
  // Set from --path-cov-instrument-only; empty when unset, and then this
  // narrowing does not exist and behaviour is bit-identical to before.
  //
  // WHY IT IS A SECOND OPTION AND NOT A CHANGE TO THE FIRST. `--focus-function`
  // answers two different questions with one value, and they are only the same
  // question by coincidence:
  //
  //   * WHICH ENTRIES THE HARNESS MAY CALL -- the dispatcher alphabet
  //     (solidity_convert_constructor.cpp's `get_unbound_function`);
  //   * WHICH UNITS ARE ENUMERATED AND INSTRUMENTED -- the published
  //     denominator (the narrowing just above).
  //
  // A state-guarded path needs the first to be WIDE (someone has to establish
  // the state; one transaction is exactly one entry call, so a second letter in
  // the alphabet is the only way another unit ever runs) and the second to stay
  // NARROW (the unit under test is one unit, and its denominator must not move
  // or the ladder's cells stop being comparable).
  //
  // MEASURED, and this option exists because of it. aqua `dock` at
  // `--focus-function dock,ship`: `dock` alone enumerates 63 paths, `ship`
  // enumerates 2733, and the run instrumented 2796 and was killed at the 300 s
  // outer timeout with no usable answer -- twice, at tx=1 and tx=2. The
  // widening that was supposed to buy ONE extra caller bought the widest unit
  // in the contract as well.
  //
  // ---- THE DIRECTION THAT IS SAFE, AND THE ONE THAT IS NOT ----
  //
  // `util/config/focus_function.h` argues that the dispatcher filter and the
  // instrumentation filter must not disagree, and it is right about the
  // direction it names: a unit the dispatcher CAN enter but nothing
  // instruments is an invisible zero in the numerator. This option is the
  // OPPOSITE containment and it is checked, not assumed:
  //
  //     instrument_only  SUBSET OF  focus_function
  //
  // so every instrumented unit is still enterable. The dispatch site REFUSES a
  // value that is not a subset rather than silently intersecting -- an
  // instrumented unit the harness cannot enter would report every one of its
  // paths `unit-not-entered`, which reads as "nothing reaches this code" and is
  // actually "nothing was asked to".
  //
  // What it does NOT change: the expansion loop, the ABI gate, Phase-1 `tr`
  // accounting, both censuses. Same reasoning as the focus narrowing above --
  // the instrumented unit's body, and therefore every `enc`, every depth and
  // every stable path id, stays bit-identical to a whole-contract run's.
  std::string instrument_only = "";

  // Does `--focus-function focus` select the unit `unit_id`?
  //
  // ONE matcher, used by both the instrumentation narrowing and
  // audit_entry_liveness, so the two can never disagree about which unit the
  // focus names -- a drift that would classify a focused-but-never-entered unit
  // as "excluded by design" (informational) instead of as the hard failure it
  // is.
  //
  // Unit ids are `sol:@C@<C>@F@<fn>#<node-id>`, so the test is on the <fn>
  // segment, plus the fully mangled spelling for callers that have one.
  //
  // EXACT equality on <fn>, deliberately, because that is precisely what the
  // frontend's dispatcher filter tests (`func_name != focus_func`, on the
  // source-level name from funcSignatures). Matching the same set matters in
  // both directions: name-only matching keeps EVERY OVERLOAD, which the
  // dispatcher also offers, so no entry can be entered without being measured.
  //
  // NO `<focus>_` PREFIX RULE, although `--function` has one (is_target_func)
  // for modifier renaming. MEASURED on regression solidity_path_cov_modifier_-
  // expands: with a modifier the UNIT keeps the source name
  // (`sol:@C@M@F@set#38`) and the renamed body `set_onlyOwner` is the NON-unit
  // that gets expanded into it -- so the prefix rule buys nothing here and would
  // over-match a sibling public function whose name merely starts with
  // `<focus>_`.
  static bool
  focus_selects_unit(const std::string &unit_id, const std::string &focus);

  // --cov-report-json: the report exists FOR the per-path counterexample
  // payload (call arguments / EVM environment / post-state). The symex slicer
  // keeps only what the CLAIM depends on, and a path claim's guard mentions
  // nothing but the ghost accumulators `tr`/`cnt` — so every contract-state
  // write and every environment read is sliced out and the payload comes back
  // empty. When this is set, solidity_path_coverage() registers exactly the
  // symbols the harvest reads (contract instance objects, contract-scope
  // mapping/dynamic-array stores, msg./tx./block.) in config.no_slice_names,
  // ESBMC's existing per-symbol slicing exemption. Slicing itself stays ON, so
  // the c2goto crypto/ABI tables and the rest of the harness plumbing are
  // still removed from the formula.
  bool protect_ce_symbols = false;

  // Write the counterexample journal (see path_ce_journal_path). Its own flag
  // rather than a reuse of protect_ce_symbols: both are set by
  // --cov-report-json today, and both would keep working if one of them moved,
  // which is exactly how a mechanism ends up silently disabled by a change to
  // an unrelated flag.
  bool emit_ce_journal = false;

  // Record the per-path decision sequence (see path_decision_table). Its own
  // flag rather than a reuse of protect_ce_symbols: both happen to be set by
  // --cov-report-json today, but they are different obligations, and keying a
  // recorder off an unrelated flag's value is how a mechanism ends up silently
  // disabled by a change to that other flag.
  bool emit_decision_sites = false;

  // ---- STAGE 2: THE CERTIFICATION QUERY (--path-cov-certify <json>) ----
  //
  // Everything downstream of path enumeration rests on ONE query:
  //     assume(L <= x <= U);  assert(tr == pi)
  // "every input in the box walks path pi". It is what makes certification
  // immune to goal-cap truncation (the cap drops exit asserts, never the
  // Phase-1 `tr` accounting, so an input reaching a dropped path still carries
  // that path's number and the query rejects the box). Until now that argument
  // lived only in a comment — the query itself had no implementation at all,
  // which made it the one contributed mechanism no code had ever exercised.
  //
  // THE ASSERT GOES ON EVERY EXIT OF THE UNIT, not on pi's exit. An input in
  // the box that walks a DIFFERENT path leaves through a different exit; with
  // the assert only on pi's exit it would never be checked and the query would
  // hold vacuously — a permanently green check in the single place where being
  // green must mean something. On every exit, that same input hits
  // `tr == enc` at ITS exit, fails, and the counterexample IS the witness that
  // shrinks the box.
  //
  // Deliberately implemented as a branch at the END of solidity_path_coverage()
  // rather than a pass of its own: expansion, the ABI gate, Phase-1 accounting,
  // the `tr`-completeness invariant and both censuses must all still run. They
  // are the defences, and a certification mode that bypassed them would be
  // certifying against accounting nobody checked. It also matters that the query
  // uses the SAME `tr` the enumeration uses — if the two used different
  // accounting, the immunity-to-truncation argument would not hold.
  //
  // Empty => disabled, and the pass behaves exactly as before.
  std::string path_cov_certify_path = "";
  // Spec for the outer-box batch (see report_outer_boxes). Empty => disabled.
  std::string path_cov_outer_box_path = "";

  // ---- STAGE 3: POST-STATE ASSERTION SYNTHESIS (--path-cov-assert <json>) ----
  //
  // The certification query says WHICH inputs walk a path. It says nothing
  // about what that path DOES, and a generated test needs both: the region
  // becomes the test's `require`, and the post-state assertion becomes its
  // `assertEq`. This mode synthesises the second half and certifies it under
  // the first: assume the region at unit entry, then assert each candidate at
  // THAT path's own exit under the path-identity antecedent
  // `tr != enc || cnt != depth`, so every candidate is vacuous on every other
  // path and the whole ladder is judged in one run.
  //
  // Only SIGNS and BOUNDS are ever emitted (post == pre, post != pre,
  // post >= pre, ..., an interval on post, a bounded delta). There is
  // deliberately no `post == <model value>` rung: a model value is a fact about
  // one counterexample, and asserting it would produce a test that is red on
  // any input the region admits but the solver did not pick.
  //
  // Empty => disabled, and the pass behaves exactly as before.
  std::string path_cov_assert_path = "";

  // Set by solidity_path_coverage() when a certification query was emitted, so
  // the reporting side can behave differently WITHOUT re-reading the CLI.
  static bool path_cov_certify_mode;
  // The coordinate names the box bounds. Kept because the witness audit below
  // needs to know what a refutation is obliged to report.
  static std::vector<std::string> path_cov_certify_box_names;
  // The box being certified, and the path's own counterexample if the driver
  // supplied it. Kept so that a REFUTATION can be turned into the next box to
  // try instead of just a verdict: the witness is the input inside the box that
  // leaves the path, so the box has to be cut on the witness's side, and the
  // path's counterexample is what says which side that is. Without both, the
  // loop has to fall back to blind bisection, which is the search that was
  // withdrawn.
  static std::vector<std::array<std::string, 3>> path_cov_certify_box;
  static std::map<std::string, std::string> path_cov_certify_ce;

  // ---- THE CERTIFY-SIDE NON-VACUITY WITNESS ----
  //
  // The stage-3 ladder got this defence first; certification, which is the
  // OLDER and more quoted of the two, had none. The hole is identical and the
  // consequence is worse, because a certification run's whole output is one
  // verdict line: the four structural gates in front of the box are all
  // SYNTACTIC (lo > hi, a name bounded twice, holes emptying the interval, a
  // decimal outside the coordinate's type) and none of them can see that the
  // box is unsatisfiable SEMANTICALLY. Contract state is not havoc'd at
  // `--solidity-max-tx 1`, so `state.x in [0,0]` against a constructor that
  // assigns 7 is well-formed, in-type, non-empty -- and admits no execution at
  // all. Every exit assert then holds for want of an execution and the run
  // prints VERIFICATION SUCCESSFUL with exit 0, which is a FALSE certificate
  // rather than a weak one.
  //
  // The witness is one extra claim at pi's OWN exit carrying only the path
  // identity antecedent `tr != enc || cnt != depth`. It is REFUTED exactly when
  // some execution the box admits walks THIS path -- the property the whole
  // certificate is conditioned on. At ENTRY it would only witness that the unit
  // is reachable, which passes on a box that reaches the unit and never this
  // path.
  //
  // It is deliberately NOT one of the `#exitN` claims: those are the
  // certificate, this is its precondition, and folding it in would make a
  // vacuous run indistinguishable from a refuted one.
  static std::pair<std::string, std::string> path_cov_certify_nonvacuous_key;

  // The `#exitN` claim keys this mode emitted, in emission order. Kept because
  // the RESULT line has to tell REFUTED from VACUOUS, and "some claim was
  // refuted" is not enough: the non-vacuity witness is refuted on every
  // SUCCESSFUL certification, so a reader of `all_claims` alone would call
  // every certificate a refutation.
  static std::vector<std::pair<std::string, std::string>>
    path_cov_certify_exit_keys;

  // Print `--path-cov-certify: RESULT: CERTIFIED | REFUTED | VACUOUS`.
  //
  // THE RUN'S OWN VERDICT LINE IS NOT THE RESULT OF THIS MODE, and that is a
  // consequence of the witness above rather than a preference: the witness is
  // REFUTED on a successful certification, so the run prints VERIFICATION
  // FAILED for a box that certified. A driver reading the verdict line would
  // then read every certificate as a refutation -- so the tool states its own
  // result on a line of its own, and the driver reads THAT.
  //
  // The two changes are inseparable. Emitting the witness without the RESULT
  // line silently inverts every certification the driver has ever recorded;
  // emitting the RESULT line without the witness leaves the vacuity hole open
  // while looking as though it had been closed.
  static void report_path_cov_certify();

  // ---- PUNCHED INTERVALS (Definition 5): R_c = [L, U] \ H ----
  //
  // The values REMOVED from each coordinate's interval. A closed interval alone
  // cannot express `everything except v`, and that is not a cosmetic loss — it
  // makes the YIELD depend on an arbitrary solver choice. MEASURED, same
  // contract, same coordinate, same probes, changing only WHICH counterexample
  // the sibling happened to return (both legal members of its domain):
  //
  //     sibling CE = 2^160-1  ->  region `to in [256, 2^160-1]`  (~1.46e48)
  //     sibling CE = 0        ->  region `to in [0, 254]`        (255)
  //
  // Both are correct — each is a subset of the true domain
  // `[0,254] U [256, 2^160-1]` — and they differ by a factor of 5.7e45. The
  // subtraction can only keep the side that contains its own counterexample,
  // so the side it keeps is decided by a value nobody chose. With a hole, BOTH
  // cases produce `[0, 2^160-1] \ {255}` and the counterexample stops mattering.
  //
  // A hole is legal under exactly the rule the side cuts already obey: it must
  // keep a KNOWN member of this path's domain (its own counterexample), and it
  // only ever makes the region SMALLER. What changes is the cost — removing one
  // value instead of a whole side.
  static std::map<std::string, std::vector<std::string>> path_cov_certify_holes;

  // A REFUTATION WITHOUT A WITNESS IS WORTHLESS — hard-fail on one.
  //
  // The verdict and the witness fail INDEPENDENTLY, which is what makes this
  // worth an invariant rather than a comment. Measured: decorating the claim
  // comment with a `certify:` prefix left the report's `path_function` reading
  // `certify:sol:@C@Box@F@f#18`, the counterexample harvest builds the expected
  // argument scope from that string, every nondet failed the scope test and was
  // filed as harness-internal (dropped 19 -> 25) — and the run still printed a
  // perfectly correct FAILED verdict with an empty `inputs`.
  //
  // The damage is entirely downstream: the refuting INPUT is what the box gets
  // shrunk with, so a witness-less refutation leaves the generalisation loop
  // with nothing to do. Its symptom would be "the loop does not converge" or
  // "the loop collapses to a point", several hundred lines away from a string
  // prefix. Checking it here costs one comparison; finding it there costs hours.
  //
  // `ce_payload_requested` narrows the check to where its premise holds. The
  // counterexample payload is only harvested when the report asks for it; a run
  // that never asked has no witness to be missing, and demanding one there is
  // the audit accusing the tool of a defect it does not have. MEASURED on this
  // audit's very first real use — the same false positive the entry-liveness
  // audit produced on ITS first use (with --focus-function), and the same fix:
  // narrow to where the premise holds rather than weaken the check everywhere.
  static void audit_certify_witness(bool ce_payload_requested);

  // ---- STAGE 2, step 1: THE OUTER BOX (--path-cov-outer-box <json>) ----
  //
  // The certification query answers "is this box inside D_pi?". It does not say
  // where to look, and the way NOT to find out is to widen from the CE point and
  // ask again each time — that is a search, it has no terminating condition, and
  // it sits exactly on the "too coarse fails / too fine is expensive" dilemma
  // that got the widening route withdrawn.
  //
  // Instead measure the OUTER box first: `assume(tr == pi); assert(temp_c <= U)`
  // says D_pi is CONTAINED in the box. Assumption fixed, assertions varied — so
  // an entire ladder of candidate bounds, for every coordinate and every path,
  // is judged in ONE run. Then the certified region is computed by SUBTRACTING
  // the siblings' outer boxes, which costs no query at all: path domains
  // partition the input space, so an input in nobody else's outer box must walk
  // this path.
  //
  // The ladder's SPAN comes from the sibling counterexamples — a sibling CE at
  // value v proves the boundary lies between this path's CE and v, so the probes
  // go there instead of across the whole 256-bit type. The tool does not compute
  // that span: the driver passes it in, keeping the method abstract and the tool
  // free of any dependence on its own report format.
  //
  // Resolution is stated honestly rather than implied: K non-adaptive probes in
  // one batch give resolution (hi-lo)/(K+1), NOT log(hi-lo) — that would need K
  // adaptive rounds. Refining is the driver's job, and each round is one more
  // batch.
  struct outer_box_probet
  {
    uint64_t enc = 0;
    std::string coord;
    bool upper = false; // true: `temp_c <= value`; false: `temp_c >= value`
    std::string value;
    std::pair<std::string, std::string> key; // claim key, to read the verdict
  };
  static bool path_cov_outer_box_mode;
  static std::vector<outer_box_probet> path_cov_outer_box_probes;
  // Every path enumerated for the target unit, as (enc, depth). Needed because
  // the subtraction is over SIBLINGS, so a path that got no probes still has to
  // be visible as "unmeasured" rather than silently treated as absent.
  static std::vector<std::pair<uint64_t, uint64_t>> path_cov_outer_box_paths;
  // (enc, coordinate) -> that path's counterexample value on it. Supplied by the
  // driver from the enumeration run. Needed because a subtraction cut is legal
  // only if it keeps a KNOWN member of the path's domain; without one there is
  // nothing to stop the greedy cut from carving away the real region.
  static std::map<std::pair<uint64_t, std::string>, std::string>
    path_cov_outer_box_ce;
  // Each coordinate's own TYPE range, as decimal strings. This is an outer
  // bound for free — every value of a `uint256` is in [0, 2^256-1] — and taking
  // it is not an optimisation but a correctness fix: a probe like `a >= 0` on an
  // unsigned type is a tautology, gets simplified out of the formula, and comes
  // back with NO verdict. Read as "this bound was not established" it leaves the
  // coordinate half-open and blocks the subtraction entirely, which is exactly
  // what happened before this was added. Probes then only ever TIGHTEN it.
  static std::map<std::string, std::pair<std::string, std::string>>
    path_cov_outer_box_type_range;
  // Coordinates pinned for this batch. Every measured and every subtracted
  // region is a statement about the SLICE through these values, so they are
  // printed with the region — a region measured under `bal == 0` and rendered
  // without it would be a claim about inputs that were never examined.
  //
  // Holds the pins that were ACTUALLY APPLIED. A pin naming a coordinate the
  // tool cannot express is dropped from here (and recorded below) rather than
  // left in, because this list is what the region is LABELLED with: keeping an
  // unapplied pin would print "measured under state.s == 0" for a measurement
  // in which nothing constrained `state.s`.
  static std::vector<std::pair<std::string, std::string>>
    path_cov_outer_box_pins;

  // ---- S2: the measured unit is a NAMED OBSTACLE ----
  //
  // Non-empty names the reason. The outer box is a MEASUREMENT, not a
  // certificate, so unlike certify this does not refuse the round -- a
  // containment statement about an obstructed unit is still a true containment
  // statement. What it must not do is travel WITHOUT the caveat: every region
  // line the subtraction prints is a candidate a driver hands straight to the
  // certification query, and on an obstructed unit that query can answer
  // SUCCESSFUL about executions the chain does not have.
  //
  // Recorded here rather than read from `named_obstacle_paths` at report time,
  // because that map is filled by the insertion loop the outer-box branch
  // `continue`s past: in this mode it is EMPTY, and a reader of it would print
  // no caveat while looking exactly like a reader that had checked.
  static std::string path_cov_outer_box_obstacle;

  // ---- Coordinates the tool REFUSED to express, and why ----
  //
  // Two things can make a coordinate unusable, and they used to be one thing:
  // an ABORT. The name may not resolve (a mapping, a field access like
  // `immutables.taker`), or it may resolve to a value this stage cannot bound
  // (a string, a contract/interface handle, an aggregate — the SMT layer then
  // dies with "Projecting from non-tuple based AST" or a "Tuple AST mismatch"
  // assertion, i.e. a core dump instead of a recorded failure). Measured five
  // times across three projects and three different types; unrecognised types
  // are unbounded in number, so the rule is REFUSE BY DEFAULT with a whitelist
  // of what can be bounded, not accept-by-default with a crash on the unknown.
  //
  // Recorded rather than silently omitted. In the query, "the box omits c" and
  // "c is unconstrained" are the SAME constraint — so a refused coordinate that
  // simply vanished from the report would read as "measured, and it came out as
  // the whole type", which is a claim about a measurement that never happened.
  //
  // The refusal is NOT symmetric between the two stages, and the asymmetry is
  // the point:
  //   * OUTER BOX — refuse the coordinate, keep measuring the others. The box
  //     is a containment statement per coordinate; one missing coordinate costs
  //     information, not correctness. It must never be treated as a measured
  //     `[0, TYPE_MAX]` bound for THIS path, which would widen this path's own
  //     region and break the only-ever-narrower invariant.
  //   * CERTIFY — refuse the QUERY. Dropping a requested bound and answering
  //     SUCCESSFUL would certify a WIDER box than the one asked for, which is
  //     the single outcome that query exists to prevent.
  static std::map<std::string, std::string> path_cov_refused_coords;

  // Read the probe verdicts, print each path's outer box, then subtract the
  // siblings' boxes and print the certified region. Called after solving.
  static void report_outer_boxes();

  // ---- STAGE 3: the candidate ladder and how its verdicts are read back ----
  //
  // One record per emitted assertion. `key` is the (comment, location) pair
  // that multi_property_check files the verdict under, so the reporter reads
  // exactly the claims this mode created and nothing else.
  //
  // `rung` and `var` together are unique BY CONSTRUCTION, and that is load
  // bearing rather than tidy: `all_claims` is a std::set of (comment,
  // location), so two candidates sharing a comment at one location silently
  // collapse into one claim -- which reads downstream as a candidate that was
  // never asked about, not as one that was lost. The emitter asserts the
  // uniqueness instead of relying on it (same lesson as the outer box's probe
  // de-duplication).
  struct assert_candidatet
  {
    uint64_t enc = 0;
    std::string var;  // state variable base name
    std::string rung; // "eq"|"ne"|"ge"|"le"|"gt"|"lt"|"abs"|"delta"
    std::string text; // human-readable candidate, e.g. "post >= pre"
    std::pair<std::string, std::string> key; // claim key, to read the verdict
  };
  static bool path_cov_assert_mode;
  static std::vector<assert_candidatet> path_cov_assert_candidates;

  // ---- THE NON-VACUITY WITNESS, and why the ladder is worthless without it --
  //
  // The region is ASSUMED at unit entry. Every structural gate on it is
  // SYNTACTIC -- lo > hi, a name bounded twice, holes that empty the interval,
  // a decimal outside the coordinate's type. None of them can see that the
  // region is unsatisfiable SEMANTICALLY, and that is the common case rather
  // than an exotic one: contract state is not havoc'd at `--solidity-max-tx 1`,
  // so `state.x in [0,0]` against a constructor that assigns 7 is well-formed,
  // in-type, non-empty, and admits no execution at all. An unsatisfiable
  // assumption makes EVERY candidate hold for want of an execution, and the
  // ladder then prints a full set of certified post-state assertions about a
  // region that contains nothing.
  //
  // So one extra claim is emitted at pi's OWN exit carrying only the antecedent,
  // `tr != enc || cnt != depth`. It is REFUTED exactly when some execution
  // admitted by the region walks THIS path -- which is the property the whole
  // ladder is conditioned on. Placing it at ENTRY instead would only witness
  // that the unit is reachable, which is a strictly weaker statement and would
  // pass on a region that reaches the unit but never this path.
  //
  // Kept out of `path_cov_assert_candidates` on purpose: it is a precondition
  // of the table, not a row of it, and counting it would make every ladder
  // summary read one REFUTED too many.
  static std::pair<std::string, std::string> path_cov_assert_nonvacuous_key;

  // Print the per-candidate verdict table. Called after solving, INSTEAD of the
  // [Coverage] block: in this mode a claim that HOLDS is the wanted outcome, so
  // the coverage counters would report a completely successful ladder as 0%.
  //
  // Order is the emission order (state variables in the contract object's own
  // component order, rungs in a fixed order), never completion order, so the
  // printed table is identical under --parallel-solving.
  static void report_path_cov_assertions();

  // Path to the cross-run covered-set JSON (--coverage-covered-set).
  // Empty => disabled (no load, no skip, no write-back; behaviour
  // identical to before Item 2). Read at branch_coverage() entry.
  std::string covered_set_path = "";

  // Item 5-d: contracts whose own decisions are excluded from branch
  // coverage entirely (--coverage-exclude-contract, repeatable). A
  // decision whose "sol_decl_contract" is in this set is dropped BEFORE
  // all_claims.insert, so it counts in NEITHER the denominator NOR the
  // numerator (true dependency exclusion, e.g. OpenZeppelin under
  // --coverage-whole-unit). Empty => no exclusion. In default
  // per-contract mode scope_contract already filters foreign decisions,
  // so this set is a no-op there by construction.
  std::set<std::string> exclude_contracts;

  // k-path coverage knobs (see #4325 "Decided defaults").
  // n  : prefix depth — number of consecutive branches in each witness
  //      (default 4 if --unwind is unset, else --unwind).
  // d  : post-simplification depth cap on the witness expression tree.
  //      Witnesses deeper than d are skipped in Phase 1 (no ghost-flag
  //      fallback yet — see #4325).
  // m  : per-function goal cap. On overflow, instrumentation aborts with
  //      an actionable error rather than silently truncating.
  size_t k_path_n = 4;
  size_t k_path_witness_depth = 8;
  size_t k_path_max_goals = 10000;

protected:
  expr2tc handle_single_guard(const expr2tc &expr, bool top_level);
  void handle_operands_guard(
    const expr2tc &expr,
    goto_programt &goto_program,
    goto_programt::instructiont::targett &it);
  void add_cond_cov_assert(
    const expr2tc &top_ptr,
    const expr2tc &pre_cond,
    goto_programt &goto_program,
    goto_programt::instructiont::targett &it);
  void gen_cond_cov_assert(
    const expr2tc &top_ptr,
    const expr2tc &pre_cond,
    goto_programt &goto_program,
    goto_programt::instructiont::targett &it);

  // Foundry revert fidelity: conservative straight-line walk from an edge's
  // first instruction. Returns true iff the edge unconditionally reaches a
  // revert terminator (a FUNCTION_CALL to a `#sol_error` function — a lowered
  // `revert CustomError(...)`) before any control-flow change (goto/return/
  // end/throw/catch) or downstream merge, so a nested conditional revert never
  // taints the enclosing edge. Used by branch_coverage() to stamp the reverting
  // edge's probe with `sol_revert_edge`; the Foundry generator reads that off
  // the covered claim to emit `vm.expectRevert()`.
  bool edge_reaches_error_revert(
    goto_programt::const_targett it,
    goto_programt::const_targett end) const;

  namespacet ns;
  goto_functionst &goto_functions;

  // we need to skip the conditions within the built-in library
  // while keeping the file manually included by user
  // this filter, however, is unsound.. E.g. if the src filename is the same as
  // the builtin library name
  std::string filename;

  int target_num;
};

class goto_coverage_rm : goto_convertt
{
public:
  goto_coverage_rm(
    contextt &_context,
    optionst &_options,
    goto_functionst &goto_functions)
    : goto_convertt(_context, _options), goto_functions(goto_functions)
  {
    options.set_option("goto-instrumented", true);
  }
  void remove_sideeffect();
  goto_functionst &goto_functions;
};
