#ifndef CPROVER_GOTO_SYMEX_FOUNDRY_H
#define CPROVER_GOTO_SYMEX_FOUNDRY_H

#include <goto-symex/equation/symex_target_equation.h>
#include <solvers/smt/smt_conv.h>
#include <util/symtab/namespace.h>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

/// Generates Foundry (`*.t.sol`) test-cases for Solidity contracts.
///
/// A Solidity coverage/counterexample run drives contract methods through the
/// multi-transaction dispatcher with nondet parameters. Since the fix that
/// makes scalar harness parameters flow as `nondet$symex` symbols
/// (solidity_convert_call.cpp assign_param_nondet), each param value is
/// recoverable from the SMT model AND its SSA assignment `original_lhs` names
/// the owning method+parameter (`sol:@C@<C>@F@<method>@<param>`). This
/// generator reconstructs the transaction sequence from those assignments and
/// emits a compilable Foundry test that replays it, mirroring the
/// ctest/pytest generators used for C/Python.
class foundry_generator
{
private:
  /// One reconstructed argument: its Solidity source type (from the
  /// `#sol_type` irep on the parameter symbol) and the formatted literal.
  struct sol_arg
  {
    std::string param;    // parameter name
    std::string sol_type; // e.g. "UINT256", "ADDRESS", "BOOL"
    std::string literal;  // Solidity literal text, or empty if unformattable
    expr2tc value;        // raw recovered value (nil if not recovered)
    // Interface-arg mock synthesis: when this argument is an interface/abstract
    // contract handle (`#sol_type: CONTRACT`), `mock_iface` names the interface
    // and a deployed `ESBMCMock_<iface>` instance is passed instead of a
    // literal. `mock_key` (the parameter name) gives each interface argument its
    // OWN mock instance (fresh-per-slot), so two handles a constructor forces
    // distinct (`a != b`) get distinct deployed addresses. Empty otherwise.
    std::string mock_iface;
    std::string mock_key;
    // True when the literal is a type DEFAULT (0/address(0)/…) because no value
    // was recovered for this parameter and it is not an interface mock. A
    // defaulted constructor argument may violate a ctor `require` and revert
    // setUp, so a deploy with any defaulted arg is degraded to UNSUPPORTED.
    bool defaulted = false;
    // Extra top-level types a rendered struct literal references and that must
    // be imported (field UDVT names, nested struct scopes) — the struct's own
    // scope is carried via `sol_type` ("STRUCT:<Qualified>").
    std::set<std::string> struct_imports;
  };

  /// One reconstructed external call `c.method(args)`.
  struct sol_call
  {
    std::string contract;
    std::string method;
    std::vector<sol_arg> args;
    bool supported = true; // false if any arg type could not be formatted
    bool reverts = false;  // covered edge reverts -> wrap in vm.expectRevert()
    // The covered path's exit is CONFIRMED normal, so the call is emitted bare
    // -- no try/catch. That bareness IS the assertion: a revert at run time
    // fails the test. Distinct from `!reverts`, which until now meant only
    // "not confirmed to revert" and was emitted revert-tolerantly, i.e. with no
    // assertion at all.
    //
    // The three states must stay three. Collapsing "confirmed normal" into
    // "not confirmed to revert" is what made every generated test
    // assertion-free, and collapsing "not confirmed" into "normal" would assert
    // that a reverting transaction succeeded. Only complete-path coverage
    // supplies the distinction (goto_coveraget's revert / rollback_revert /
    // undetermined_exit sets); branch coverage has no third state and leaves
    // this false, keeping its output byte-identical.
    bool normal_confirmed = false;
    // This call's covered path is a NAMED OBSTACLE: the model admits an
    // execution the chain does not have, so the counterexample behind it may
    // describe something that cannot happen and a test built from it is RED on
    // the UNMODIFIED contract.
    //
    // goto_coverage.h states the rule outright -- "a marked path ... must not be
    // turned into a test. Marking without excluding would be worthless" -- and
    // the marking half was the only half implemented. MEASURED:
    // named_obstacle_paths has exactly three readers and all three sit under
    // `if (tri == "U")` in bmc.cpp, so a REFUTED path never reaches any of them.
    // Refuted is the only kind that ever becomes a test. The regression
    // solidity_path_cov_residual_unit_call_obstacle pins 5 obstacle paths of
    // which just ONE shows as `named-obstacle` in the report -- the other four
    // are F, and flowed straight through to this generator.
    //
    // Set from the SAME (comment, location) key the census stores, alongside the
    // normal_exit_paths lookup, rather than by matching the prettified claim text
    // later: a detector keyed on a string that does not round-trip is a detector
    // that never fires, which is the failure this is fixing, one layer down.
    bool named_obstacle = false;
    // ③A0 environment pinning: the msg.value the solver picked for this call's
    // transaction (recovered from the per-tx `_sol_per_tx_reseed`). Emitted as
    // `{value: N}` ONLY when `payable` — sending value to a non-payable method
    // reverts. Nil / non-payable -> no value pin.
    expr2tc msg_value;
    bool payable = false;
    // ③A0 block.timestamp pinning: the timestamp the solver picked, emitted as
    // `vm.warp(N)` ONLY when `warp` (the covered path actually reads
    // block.timestamp) — otherwise a nondet timestamp is noise that can spuriously
    // revert unrelated arithmetic. Nil / !warp -> no warp.
    expr2tc block_timestamp;
    bool warp = false;
    // ③A0 msg.sender pinning (deployer coordination): the per-tx sender the
    // solver picked, emitted as `vm.prank(N)` ONLY when `prank` — the covered
    // path reads msg.sender AND the segment is sender-clean (no nested-call
    // wrapper overwrote the top-level sender, so `vm.prank` faithfully
    // reproduces it). A ctor carrier instead uses `deployer` to emit a setUp
    // `vm.startPrank(N)` around `new C()`, making `owner = msg.sender` deploy
    // under a known identity so an `onlyOwner` call can match (or mismatch) it.
    expr2tc msg_sender;
    bool prank = false;
    bool deployer = false;
    // Set on a ctor carrier that had to be marked unsupported because the model's
    // deploy path needs a nonzero msg.value the (non-payable) constructor cannot
    // legally receive — used only to emit an accurate UNSUPPORTED reason.
    bool ctor_value_unsendable = false;
    // Set on a deploy-time env carrier for a PARAMETERIZED constructor whose
    // arguments were not recovered (e.g. `--focus-function` mode nondets them):
    // the deploy is marked UNSUPPORTED rather than emit all-default args, which
    // could revert setUp (a require on a zero default) and break the whole suite.
    bool ctor_unrecovered = false;
    // Set on a deploy ctor whose args were recovered under a BASE contract and
    // remapped onto the deploy contract (empty derived body forwarding to a base
    // ctor, e.g. EscrowDst -> BaseEscrow). Reported for visibility.
    bool ctor_remapped = false;
  };

  /// One counterexample -> one test function: a sequence of calls.
  using test_case = std::vector<sol_call>;

  std::vector<test_case> test_cases;
  std::string source_file;
  mutable std::mutex data_mutex;

  /// Counterexamples REFUSED because their path is a named obstacle (see
  /// sol_call::named_obstacle). Counted rather than silently dropped: a
  /// suppression that leaves no trace is indistinguishable from a path that was
  /// never witnessed, and the whole point of the obstacle machinery is that an
  /// excluded path is excluded VISIBLY. Reported by generate() next to the
  /// bare/try-catch counts, in the same absolute-number style the obstacle
  /// warning already uses -- an obstacle is not partial credit.
  size_t suppressed_obstacle = 0;

  /// Counterexamples REFUSED because every call they reconstructed is a
  /// CONSTRUCTOR, so the test function they would produce has an EMPTY BODY.
  ///
  /// MEASURED end to end on aqua: `dock` and `ship` each emitted
  ///
  ///     // claim: sol:@C@Aqua@F@dock#3088:path:12, ...:path:2
  ///     function test_cov_0() public {
  ///     }
  ///
  /// naming two witnessed paths and executing neither, and `forge test`
  /// reported `[PASS] test_cov_0() (gas: 188)` -- it passes BECAUSE it does
  /// nothing. That is worse than a missing test: it is counted as emitted,
  /// counted as passing, and the only thing distinguishing it from a real test
  /// is reading the body. Two of the six files the aqua round-trip produced
  /// were of this shape.
  ///
  /// The cause is structural rather than exotic: the emission loop skips a call
  /// whose method IS its contract (`continue; // constructor -> setUp()`), so a
  /// case that reconstructed only a constructor segment leaves nothing behind.
  ///
  /// Refused in collect(), beside the obstacle refusal and for the same stated
  /// reason: the case must never enter `test_cases`, or dedup can collapse it
  /// onto a legitimate case by fingerprint and ship one under the other's
  /// provenance. Counted, and reported by generate() -- a silent refusal is
  /// indistinguishable from a path that was never witnessed.
  size_t suppressed_empty_body = 0;

  /// WHY the body was empty. `suppressed_empty_body` says a case reconstructed
  /// only a constructor; it does not say which of two very different things
  /// happened, and the two need opposite fixes:
  ///
  ///   (a) a call to the unit WAS built and came back unsupported -- an argument
  ///       type that cannot be rendered. Then the fix is in the renderer.
  ///   (b) no dispatcher segment ever acquired a method, so no call was built at
  ///       all, and the coverage-claim FALLBACK that exists to repair exactly
  ///       that could not run because `calls` was already non-empty -- holding
  ///       nothing but the constructor. Then the fix is the fallback's guard.
  ///
  /// (a) is ruled out by construction on the ordinary route: an unsupported call
  /// is still pushed into `calls`, so it satisfies `method != contract` and the
  /// empty-body refusal never fires for it. But "ruled out by reading the code"
  /// is exactly the standard this project has been burned by twice -- argument
  /// aliasing and the transaction bound were both plausible, cheap, and refuted
  /// by measurement. So the distinction gets a COUNTER before it gets a fix.
  ///
  /// `segments_without_method` counts dispatcher segments dropped by the
  /// `!s.method.empty()` guard: a loop-only branch never sets the segment's
  /// method, which is the documented `dock`/`push` shape.
  /// `fallback_rescued_ctor_only` counts reconstructions in which no callable
  /// call existed but a CONSTRUCTOR had already been pushed. Under the original
  /// `calls.empty()` guard those were exactly the cases the coverage-claim
  /// fallback could not repair: they reached collect() with an empty body and
  /// were refused. The guard now asks the question the emission loop asks --
  /// is there a call that is NOT a constructor? -- so the same counter reads as
  /// how many cases the fix RESCUES.
  ///
  /// MEASURED on aqua `dock`, counter in place and guard not yet changed:
  /// `segments_without_method 0`, fallback blocked in 2 reconstructions. So
  /// there were no segments AT ALL -- per-claim slicing removes the
  /// dispatcher's first tx-guard, which is the very case the fallback exists
  /// for -- and the constructor blocked the repair. Kept after the fix rather
  /// than deleted: it is what separates this route from an unrenderable-argument
  /// route on the next benchmark, and a measurement removed once it has served
  /// one investigation makes the next one start from a guess.
  mutable size_t segments_without_method = 0;
  mutable size_t fallback_rescued_ctor_only = 0;

  /// `<C>.<m>(<param>: <sol-type>, ...)` for each call the coverage-claim
  /// fallback BUILT and then discarded as unsupported. The fallback only keeps
  /// a call when `supported` is true, so an unrenderable argument makes it a
  /// silent no-op whose only symptom is an empty-body refusal three steps
  /// later, naming neither the method nor the argument. Recorded so the next
  /// question -- renderer gap or resolution gap? -- is answered by the run.
  mutable std::set<std::string> fallback_unsupported;

  /// Contracts Solidity forbids `new` on (abstract / interface / library),
  /// detected from the `#sol_no_new` flag stamped on their constructor symbol.
  /// The generator degrades their instantiation to UNSUPPORTED so the emitted
  /// test always compiles. Populated during reconstruct() (needs the namespace).
  mutable std::set<std::string> non_instantiable;

  /// Contracts that are Solidity libraries (no `this` self-pointer on their
  /// functions). A library is called statically (`Lib.fn(args)`) and never
  /// instantiated, so it is kept out of the construction plan. Populated
  /// during reconstruct() (needs the namespace).
  mutable std::set<std::string> libraries;

  /// A synthesized mock for an interface/abstract-contract argument type. The
  /// generator emits `contract ESBMCMock_<name> is <name> { <stubs> }` so a
  /// constructor/method taking an `<name>` handle can be deployed in Foundry
  /// (a bare address would revert when the contract calls a method on it).
  struct mock_spec
  {
    std::string name;               // interface/contract type name
    std::vector<std::string> stubs; // rendered function-override lines
    // True only if EVERY function of the interface (params + returns) rendered
    // as a default-returning stub. If false, any deploy needing this interface
    // degrades to UNSUPPORTED — a partial mock would not satisfy `is <name>`
    // and would fail to compile (anti-goal: never emit an uncompilable test).
    bool renderable = false;
  };

  /// Cache of synthesized interface mocks, keyed by interface name. Populated
  /// lazily during reconstruct(); read during emission. `renderable == false`
  /// entries are memoized negatives (do not re-enumerate).
  mutable std::map<std::string, mock_spec> mock_specs;

  /// Build (or fetch cached) the mock for an interface/abstract contract:
  /// enumerate its externally-visible functions from the symbol table (inherited
  /// methods are already inlined into the derived interface's `@F@` set) and
  /// render a default-returning `override` stub for each. All-or-nothing: any
  /// unrenderable param/return type, or zero functions found, yields
  /// `renderable == false`.
  const mock_spec &
  build_mock_spec(const namespacet &ns, const std::string &iface) const;

  /// Cache of a method's declared parameters in source order:
  /// key "<contract>@<method>" -> [(param_name, sol_type)].
  mutable std::
    map<std::string, std::vector<std::pair<std::string, std::string>>>
      method_params;

  /// Look up a method's declared parameters (source order + `#sol_type`) from
  /// the symbol table; cached. Empty vector if the method is not found.
  const std::vector<std::pair<std::string, std::string>> &get_method_params(
    const namespacet &ns,
    const std::string &contract,
    const std::string &method) const;

  /// Cache of a contract's dispatcher-callable methods: base-name -> list of
  /// full method-ids (`sol:@C@<C>@F@<m>#<id>`); >1 id means the base name is
  /// overloaded. Keyed by contract name.
  mutable std::map<std::string, std::map<std::string, std::vector<std::string>>>
    dispatcher_methods;

  /// The methods a contract's `_ESBMC_Nondet_Extcall_<C>` dispatcher can
  /// invoke as a transaction entry, extracted from the dispatcher body. This
  /// is the ground truth of externally-callable methods: modifier/aux helpers
  /// (e.g. a `bump_onlyOwner` the dispatcher never calls directly) are absent,
  /// and overloads appear as multiple ids under one base name. Cached.
  const std::map<std::string, std::vector<std::string>> &
  dispatcher_callable(const namespacet &ns, const std::string &contract) const;

  /// Parse a Solidity parameter symbol name
  /// `sol:@C@<contract>@F@<method>@<param>` into its parts. Returns false if
  /// the name is not a Solidity function parameter.
  static bool parse_param_symbol(
    const std::string &name,
    std::string &contract,
    std::string &method,
    std::string &param);

  /// Model value of a focused-function parameter (contract,method,param), read
  /// from the solver for `--function` isolated-function runs (where parameters
  /// are free nondet inputs, not `param = nondet` assignments). Null expr when
  /// the parameter was sliced away (irrelevant to the covered branch).
  static expr2tc recover_focus_param(
    const symex_target_equationt &target,
    smt_convt &smt_conv,
    const std::string &contract,
    const std::string &method,
    const std::string &param);

  /// Format an SMT constant as a Solidity literal for the given `#sol_type`.
  /// Returns empty for a type we cannot faithfully render (caller then marks
  /// the call unsupported rather than emit a wrong test).
  static std::string
  format_sol_value(const std::string &sol_type, const expr2tc &value);

  /// Default literal for a supported type (used for a declared parameter that
  /// was not exercised on the path, e.g. a short-circuited operand). Returns
  /// empty for an unsupported type.
  /// `nth` disambiguates: a defaulted argument is one the path did not
  /// constrain, so any value is faithful -- but all-zero makes DISTINCT
  /// parameters into the SAME mapping key, which is an aliasing the model never
  /// implied. Non-zero `nth` shifts the identity-like types (address, bytesN)
  /// apart; numeric and boolean defaults stay at their zero, where equality
  /// aliases nothing. 0 keeps the old literal exactly, so every other call site
  /// is unchanged.
  static std::string
  default_sol_literal(const std::string &sol_type, unsigned nth = 0);

  /// Render a recovered `constant_struct` value as a Solidity positional struct
  /// literal `<Qualified>(f0, f1, …)`. Field source types (UDVT / bytesN /
  /// address / …) come from the DECLARED struct tag symbol (`tag-struct
  /// <Qualified>`), which retains them (the recovered value's migrated type is
  /// stripped); synthetic `anon_pad$*` padding components are skipped; each real
  /// field reuses format_sol_value/default_sol_literal (recursing for a nested
  /// struct field). `qualified` is set to the struct's Solidity name for import.
  /// All-or-nothing: returns "" if any field (or a fixed-array field) cannot
  /// render, so the call degrades to UNSUPPORTED rather than emit a bad literal.
  static std::string format_struct_literal(
    const namespacet &ns,
    const expr2tc &value,
    std::string &qualified,
    std::set<std::string> &out_imports);

  /// Reconstruct the ordered call sequence of one counterexample from its SSA.
  ///
  /// `out_claims` returns the VERIFICATION OBLIGATION(S) this case was
  /// reconstructed from — the guard-true coverage claims' identities, e.g.
  /// `pull:path:63`. Empty when none could be read.
  test_case reconstruct(
    const symex_target_equationt &target,
    smt_convt &smt_conv,
    const namespacet &ns,
    std::string &out_claims) const;

  /// Originating claim identity per reconstructed case, keyed by the case's
  /// dedup fingerprint.
  ///
  /// WHY THIS EXISTS. Until it did, a generated test could not say which
  /// verification obligation produced it: every case was a bare `test_cov_N`
  /// with a call and a comment. Measured consequence on aqua: the
  /// whole-contract run witnesses 15 counterexamples across 6 units and emits 4
  /// cases naming 3, with `pull` — witnessed under exactly the same path ids as
  /// the focused run — absent entirely. Whether its counterexamples were
  /// DROPPED or RENAMED as another method could not be decided from the
  /// artifact, because the artifact carries no link back to the report.
  ///
  /// The link is worth having beyond that bug. Every acceptance criterion in
  /// this project compares "what was measured" against "what was shipped", and
  /// a test that cannot name its obligation cannot be audited against the
  /// report it came from. "Verifier-derived" is a claim about provenance; this
  /// is what makes it checkable rather than asserted.
  ///
  /// Keyed by fingerprint because dedup already keys on it, so several
  /// counterexamples collapsing onto one emitted case are visible as several
  /// claims on that case rather than silently losing all but the first.
  std::map<std::string, std::string> claims_by_fingerprint;

  /// Deduplication fingerprint for a reconstructed test case.
  static std::string fingerprint(const test_case &tc);

  /// Contract under test for a case: the first non-constructor call's contract
  /// (fallback: the first call's contract). Files split one per primary.
  static std::string primary_contract(const test_case &tc);

  /// Write one `.t.sol` file for the given contract-under-test. Test cases are
  /// grouped by construction signature: each distinct construction becomes its
  /// own `contract <primary>CovTest[_n] is Test` with a `setUp()` that deploys
  /// the instance, and the cases sharing it become its `test_cov_*` functions.
  /// Returns the number of calls wrapped in revert-tolerant try/catch.
  size_t write_foundry_file(
    const std::string &path,
    const std::string &primary,
    const std::vector<test_case> &cases) const;

public:
  foundry_generator() = default;

  /// Clear collected data (start of a coverage run).
  void clear();

  /// Collect one counterexample (called per reached goal in coverage mode).
  void collect(
    const symex_target_equationt &target,
    smt_convt &smt_conv,
    const namespacet &ns);

  /// Emit the accumulated test suite.
  void generate() const;

  /// Single-shot emission for non-coverage mode.
  void generate_single(
    const symex_target_equationt &target,
    smt_convt &smt_conv,
    const namespacet &ns);

  /// Whether any test case has been collected.
  bool has_tests() const;
};

#endif
