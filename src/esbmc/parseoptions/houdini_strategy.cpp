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

  // Final run: the surviving set is inductive, so this verdict is about the
  // program's own properties.
  goto_functions = pristine;
  goto_houdini_emit_candidates(goto_functions, keep);
  goto_loop_invariant(goto_functions, context, false);

  reset_run_state();
  bmct bmc(goto_functions, options, context);
  return do_bmc(bmc);
}
