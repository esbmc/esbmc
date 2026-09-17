#include <esbmc/esbmc_parseoptions.h>
#include <esbmc/bmc.h>
#include <goto-programs/goto_houdini_invariants.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_loop_invariant.h>
#include <util/arith/arith_tools.h>
#include <goto-programs/property_verdict.h>
#include <util/base/time_stopping.h>
#include <util/message/message.h>
#include <algorithm>
#include <mutex>
#include <tuple>
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

/// The pool index a claim comment names, or nullopt when the claim is not a
/// pool candidate's (a user claim, or the `true` that only cuts a loop).
std::optional<size_t> pool_index_of(const std::string &claim)
{
  const std::string id = candidate_id_of(claim);
  if (id.empty() || !std::all_of(id.begin(), id.end(), ::isdigit))
    return std::nullopt;
  return std::stoul(id);
}

using positiont = std::tuple<std::string, unsigned, unsigned>;

positiont position_of(const property_locationt &loc)
{
  return {loc.file, loc.line, loc.column};
}

/// Where the program's own assertions are. A claim the cut program never
/// reaches has no verdict at all, so a proof must cover each of these.
std::set<positiont> assertion_positions(const goto_functionst &goto_functions)
{
  std::set<positiont> positions;
  forall_goto_functions (f, goto_functions)
  {
    if (!f->second.body_available)
      continue;
    forall_goto_program_instructions (i, f->second.body)
      if (i->is_assert())
        positions.insert(position_of(property_location(i->location, "")));
  }
  return positions;
}
} // namespace

kind_proof_resultt esbmc_parseoptionst::prove_kind_candidates(
  const goto_functionst &pristine,
  const std::vector<kind_candidatet> &pool,
  const std::vector<size_t> &ids,
  const optionst &probe_options,
  const std::map<unsigned, BigInt> &bounds,
  fine_timet deadline)
{
  const auto saved_verdicts = goto_functionst::property_verdicts.snapshot();
  const auto saved_reached = goto_functionst::reached_claims;
  const auto saved_mul_reached = goto_functionst::reached_mul_claims;

  // The schema's program with @p emitted attached. Asserting nothing but the
  // candidates keeps each refinement to the claims it is about: an assertion
  // constrains no path, so dropping one changes no reachable state.
  // Drops from @p candidates any that were not emitted.
  auto build = [&](std::vector<size_t> &candidates, bool program_claims) {
    goto_functionst probe = pristine;
    if (!program_claims)
      Forall_goto_functions (f, probe)
        Forall_goto_program_instructions (i, f->second.body)
          if (i->is_assert())
            i->make_skip();
    const std::set<size_t> emitted =
      emit_kind_candidates(probe, pool, candidates);
    candidates.erase(
      std::remove_if(
        candidates.begin(),
        candidates.end(),
        [&emitted](size_t id) { return emitted.count(id) == 0; }),
      candidates.end());
    goto_loop_invariant(probe, context, false);
    return probe;
  };

  auto solve = [&](goto_functionst &probe, bmct::kind_feedbackt &feedback) {
    std::string unwindset;
    forall_goto_functions (f, probe)
      forall_goto_program_instructions (i, f->second.body)
      {
        const auto bound = bounds.find(stamped_loop(*i));
        if (i->is_backwards_goto() && bound != bounds.end())
          unwindset += (unwindset.empty() ? "" : ",") +
                       std::to_string(i->loop_number) + ":" +
                       integer2string(bound->second);
      }

    reset_run_state();
    optionst options = probe_options;
    options.set_option("unwindset", unwindset);
    bmct bmc(probe, options, context);
    bmc.kind_feedback = &feedback;
    std::shared_ptr<symex_target_equationt> eq;
    return bmc.run(eq);
  };

  // Houdini refined by models: each satisfiable round names every candidate
  // the model breaks -- one false at an entry, or not preserved by the body
  // under the others -- and dropping one only weakens what the rest assume.
  // An unsatisfiable round has every remaining candidate, and every unwinding
  // assertion, holding together.
  kind_proof_resultt result;
  std::vector<size_t> alive = ids;
  bool inductive = false;
  while (!inductive && current_time() <= deadline)
  {
    ++result.rounds;
    goto_functionst probe = build(alive, false);
    bmct::kind_feedbackt feedback;
    const smt_resultt res = solve(probe, feedback);
    if (res == P_UNSATISFIABLE)
    {
      inductive = true;
      break;
    }
    if (res != P_SATISFIABLE)
      break;

    std::set<size_t> broken;
    for (const std::string &claim : feedback.violated)
      if (const auto index = pool_index_of(claim))
        broken.insert(*index);
    // Only an unwinding assertion failed: a loop left to the unwinder runs
    // past its bound, and no candidate can be separated from that.
    if (broken.empty())
      break;
    alive.erase(
      std::remove_if(
        alive.begin(),
        alive.end(),
        [&broken](size_t index) { return broken.count(index) != 0; }),
      alive.end());
  }

  if (inductive)
  {
    result.proven = alive;

    if (current_time() <= deadline)
    {
      ++result.rounds;
      goto_functionst probe = build(alive, true);
      bmct::kind_feedbackt feedback;
      const bool holds = solve(probe, feedback) == P_UNSATISFIABLE;

      // Every assertion the run reached holds; one it never reached has no
      // verdict, and the cut program may have made it unreachable (#7478).
      std::set<positiont> reached;
      for (const auto &[claim, verdict] :
           goto_functionst::property_verdicts.snapshot())
        reached.insert(position_of(verdict.loc));
      const std::set<positiont> assertions = assertion_positions(pristine);
      result.program_proved =
        holds && std::includes(
                   reached.begin(),
                   reached.end(),
                   assertions.begin(),
                   assertions.end());
    }
  }

  reset_run_state();
  for (const auto &[claim, verdict] : saved_verdicts)
    goto_functionst::property_verdicts.record(
      claim, verdict.verdict, verdict.loc, verdict.note);
  goto_functionst::reached_claims = saved_reached;
  goto_functionst::reached_mul_claims = saved_mul_reached;
  return result;
}

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
