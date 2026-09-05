#include <esbmc/esbmc_parseoptions.h>
#include <esbmc/bmc.h>
#include <goto-programs/goto_houdini_invariants.h>
#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/property_verdict.h>
#include <util/message/message.h>
#include <mutex>
#include <optional>
#include <set>
#include <string>

namespace
{
/// Every round but the last deletes at least one candidate, so the pool size
/// bounds the number of rounds. The cap only guards against a future change
/// that breaks that argument; hitting it is a bug, not a workload.
constexpr size_t kMaxHoudiniRounds = 32;

/// Candidate id carried by a claim comment, or empty when the claim is not a
/// Houdini candidate's. goto_loop_invariantt::invariant_claim_comment writes
/// `... [houdini-candidate:<id>]`, and the verdict table keys on that comment.
std::string candidate_id_of(const std::string &claim)
{
  const std::string open = std::string("[") + kHoudiniCandidatePrefix;
  const size_t begin = claim.find(open);
  if (begin == std::string::npos)
    return "";

  const size_t id_at = begin + open.size();
  const size_t end = claim.find(']', id_at);
  if (end == std::string::npos)
    return "";

  return claim.substr(id_at, end - id_at);
}

/// The candidates that survived the round just run: those the solver did not
/// refute. A candidate is deleted when *either* obligation failed -- a failed
/// base case means it does not hold on entry, a failed inductive step that the
/// body does not preserve it. Both make it useless, and leaving it in would
/// fail the run on a guess rather than on the program.
std::set<std::string> survivors_of_round(const std::set<std::string> &emitted)
{
  std::set<std::string> refuted;
  for (const auto &[claim, result] :
       goto_functionst::property_verdicts.snapshot())
  {
    if (result.verdict != property_verdictt::Failed)
      continue;
    const std::string id = candidate_id_of(claim);
    if (!id.empty())
      refuted.insert(id);
  }

  std::set<std::string> survivors;
  for (const std::string &id : emitted)
    if (refuted.count(id) == 0)
      survivors.insert(id);
  return survivors;
}

/// Wipe the run-scoped state that ESBMC keeps in process globals.
///
/// `reached_claims` is static, and multi-property uses it to skip claims a
/// previous run already discharged. Each Houdini round is a *different*
/// program, so carrying it over makes the final run skip the user's own
/// assertions and report them PASSED without ever solving them -- a false
/// SUCCESSFUL on a program with a real bug. The verdict table has to go with
/// it, or the skipped claims keep a stale verdict.
void reset_run_state()
{
  goto_functionst::property_verdicts.clear();
  {
    std::lock_guard<std::mutex> lock(goto_functionst::reached_claims_mutex);
    goto_functionst::reached_claims.clear();
  }
  std::lock_guard<std::mutex> lock(goto_functionst::reached_mul_claims_mutex);
  goto_functionst::reached_mul_claims.clear();
}

/// The program's own ASSERTs, before the schema rewrites anything. Invariant
/// claims are not in here: they do not exist yet.
size_t count_user_claims(const goto_functionst &goto_functions)
{
  size_t n = 0;
  forall_goto_functions (f, goto_functions)
  {
    if (!f->second.body_available)
      continue;
    forall_goto_program_instructions (i, f->second.body)
      if (i->is_assert())
        ++n;
  }
  return n;
}

/// How many of the program's own claims reached a verdict in the run just
/// finished. The invariant obligations the schema emits are excluded by their
/// property tag: they are the proof, not the thing being proved.
size_t user_claims_decided()
{
  size_t n = 0;
  for (const auto &[claim, result] :
       goto_functionst::property_verdicts.snapshot())
  {
    (void)claim;
    if (
      result.loc.description.find("loop invariant base case") ==
        std::string::npos &&
      result.loc.description.find("loop invariant inductive step") ==
        std::string::npos &&
      result.verdict != property_verdictt::NotChecked)
      ++n;
  }
  return n;
}

} // namespace

/// Houdini fixpoint: guess a pool of candidate invariants, then let the solver
/// delete the ones it refutes until the surviving set is inductive. See
/// goto_houdini_invariants.h for why the existing loop-invariant schema is
/// already the inner check.
///
/// Rounds are probes and report nothing; only the final run, made under the
/// inductive set, produces the verdict the user sees. Intermediate rounds
/// routinely fail the user's own assertions -- that is the pool still being
/// filtered, not a property violation.
int esbmc_parseoptionst::do_houdini_strategy(
  optionst &options,
  goto_functionst &goto_functions)
{
  const goto_functionst pristine = goto_functions;

  // nullopt until the first round has refuted something: the emitter then
  // emits the whole pool. An empty set means everything was refuted.
  std::optional<std::set<std::string>> keep;
  size_t rounds = 0;

  for (; rounds < kMaxHoudiniRounds; ++rounds)
  {
    goto_functionst probe = pristine;
    const std::set<std::string> emitted =
      goto_houdini_emit_candidates(probe, keep);
    if (emitted.empty())
      break;

    goto_loop_invariant(probe, context, false);

    optionst probe_options = options;
    probe_options.set_option("houdini-probe", true);

    reset_run_state();
    bmct probe_bmc(probe, probe_options, context);
    do_bmc(probe_bmc);

    const std::set<std::string> survivors = survivors_of_round(emitted);
    if (survivors.size() == emitted.size())
      break; // nothing refuted: the set is inductive

    keep = survivors;
    if (survivors.empty())
      break; // the whole pool was refuted; nothing left to assume
  }

  log_status(
    "Houdini: {} candidate(s) inductive after {} round(s)",
    keep ? keep->size() : size_t{0},
    rounds);

  const size_t user_claims = count_user_claims(pristine);

  // Final run: the surviving set is inductive, so this verdict is about the
  // program's own properties.
  goto_functions = pristine;
  goto_houdini_emit_candidates(goto_functions, keep);
  goto_loop_invariant(goto_functions, context, false);

  // The verdict is deferred to this level: do_bmc would otherwise print
  // SUCCESSFUL before the unchecked-claim guard below has run, leaving two
  // contradictory verdict lines in the output. parse_result() in
  // scripts/competitions/svcomp/esbmc-wrapper.py matches the first, so this is
  // an interface question as much as a readability one. The per-property table
  // still prints; only the one-line verdict is ours to emit.
  optionst final_options = options;
  final_options.set_option("houdini-defer-verdict", true);

  reset_run_state();
  bmct bmc(goto_functions, final_options, context);
  const int result = do_bmc(bmc);

  // The schema cuts the loop, so a claim after it is reached only through the
  // havoc-and-assume path. Where the frontend hoisted a loop guard's side
  // effects above the havoc -- `while (cnt--)` -- the exit edge tests a
  // pre-havoc temporary and is infeasible, and every post-loop claim is then
  // dropped without ever being solved (issue #7478, a --loop-invariant-check
  // defect that reproduces on master with a hand-written invariant). An
  // unreached claim is not a discharged one, so a run that lost any of the
  // program's own claims has not proved the program and must not say
  // SUCCESSFUL. Keep this guard after #7478 is fixed: it is cheap, and it
  // bounds any future way of making post-loop code unreachable.
  const size_t decided = user_claims_decided();
  if (result == 0 && decided < user_claims)
  {
    log_error(
      "Houdini: {} of the program's {} claim(s) were never checked -- the "
      "invariant schema made them unreachable. Reporting UNKNOWN rather than a "
      "proof that rests on code the run never reached.",
      user_claims - decided,
      user_claims);
    log_result("\nVERIFICATION UNKNOWN");
    return 0;
  }

  if (result == 0)
  {
    log_result("\nVERIFICATION SUCCESSFUL");
    return 0;
  }

  // Deferring the verdict skipped report_violation, and with it the mapping it
  // applies: a satisfiable answer whose only violated claims lie downstream of
  // the schema's havoc is checked against the invariant's over-approximation,
  // so it witnesses a weak guess rather than a reachable bug (issue #7480).
  // Inference makes weak guesses the common case, and without this the run
  // prints FAILED over a table reading "0 properties failed" -- which
  // parse_result() in esbmc-wrapper.py would score as a refutation.
  bmc.report_violation();
  return bmc.violation_is_abstraction_only() ? 0 : result;
}
