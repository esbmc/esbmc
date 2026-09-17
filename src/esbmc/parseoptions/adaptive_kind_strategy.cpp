#include <esbmc/esbmc_parseoptions.h>
#include <esbmc/invariant_mining.h>
#include <goto-programs/abstract-interpretation/interval_analysis.h>
#include <goto-programs/goto_invariant_synthesis.h>
#include <goto-programs/goto_k_induction.h>
#include <util/arith/arith_tools.h>
#include <util/base/time_stopping.h>
#include <util/message/message.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <string>

namespace
{
std::string
unwindset_of(const std::map<unsigned, BigInt> &bounds, uint64_t k_step)
{
  std::string set;
  for (const auto &[loop, bound] : bounds)
  {
    if (bound <= k_step)
      continue;
    if (!set.empty())
      set += ",";
    set += std::to_string(loop) + ":" + integer2string(bound);
  }
  return set;
}

bool is_power_of_two(uint64_t k)
{
  return (k & (k - 1)) == 0;
}

/// Raise each loop's bound to what the forward condition showed it needs, and
/// track the loops that can outrun the cap. Returns whether a bound changed.
bool apply_bound_hints(
  const bmct::kind_feedbackt &hints,
  uint64_t k_step,
  std::map<unsigned, BigInt> &bounds,
  std::set<unsigned> &unbounded)
{
  bool changed = false;
  for (const auto &[loop, remaining] : hints.remaining)
  {
    if (!remaining)
    {
      if (unbounded.insert(loop).second)
        log_status(
          "Loop {} can run past --kind-max-bound; relying on the inductive "
          "step",
          loop);
      continue;
    }

    unbounded.erase(loop);
    const BigInt current =
      bounds.count(loop) ? std::max(bounds[loop], BigInt(k_step))
                         : BigInt(k_step);
    bounds[loop] = current + *remaining + 1;
    changed = true;
    log_status(
      "Loop {}: forward condition bounds it at {} iterations",
      loop,
      integer2string(bounds[loop] - 1));
  }
  return changed;
}

/// Add each of @p guesses not already in @p pool, up to its loop's cap.
/// Returns whether one was added.
bool add_candidates(
  const std::vector<kind_candidatet> &guesses,
  std::vector<kind_candidatet> &pool)
{
  bool grew = false;
  for (const kind_candidatet &guess : guesses)
  {
    size_t same_loop = 0;
    bool known = false;
    for (const kind_candidatet &c : pool)
      if (c.loop == guess.loop)
      {
        ++same_loop;
        known = known || c.expr == guess.expr;
      }
    if (known || same_loop >= kMaxCandidatesPerLoop)
      continue;
    pool.push_back(guess);
    grew = true;
  }
  return grew;
}

/// Add the candidates @p samples suggest to @p pool and mark those they
/// refute. Returns whether the pool gained a candidate.
bool learn_from_samples(
  const std::vector<loop_head_samplet> &samples,
  const std::set<size_t> &assumed,
  std::vector<kind_candidatet> &pool,
  std::set<size_t> &refuted)
{
  const bool grew = add_candidates(mine_candidates(samples), pool);

  for (size_t id = 0; id < pool.size(); ++id)
    if (
      !assumed.count(id) && !refuted.count(id) &&
      refuted_by_samples(pool[id], samples))
      refuted.insert(id);
  return grew;
}

/// Add the separators between @p samples and the state each loop of an
/// inductive-step counterexample started from. Returns whether the pool grew.
bool learn_from_counterexample(
  const std::vector<loop_head_samplet> &samples,
  const std::vector<loop_head_samplet> &counterexample,
  std::vector<kind_candidatet> &pool)
{
  bool grew = false;
  std::set<unsigned> seen;
  for (const loop_head_samplet &start : counterexample)
    if (seen.insert(start.loop).second)
      grew |= add_candidates(separate_counterexample(samples, start), pool);
  return grew;
}

/// A base-case run of the loop-invariant schema, bounded at k like the base
/// case so a loop the schema leaves to the unwinder terminates.
optionst probe_options(const optionst &options, uint64_t k_step)
{
  optionst probe = options;
  probe.set_option("base-case", true);
  probe.set_option("forward-condition", false);
  probe.set_option("inductive-step", false);
  probe.set_option("k-induction", false);
  probe.set_option("no-unwinding-assertions", false);
  probe.set_option("partial-loops", false);
  probe.set_option("no-assertions", false);
  probe.set_option("multi-property", false);
  probe.set_option("unwind", std::to_string(k_step));
  probe.set_option("unwindset", "");
  return probe;
}

/// @p bounds, keyed by loop stamp rather than by the loop number that the
/// schema's program no longer shares with @p goto_functions.
std::map<unsigned, BigInt> bounds_by_stamp(
  const goto_functionst &goto_functions,
  const std::map<unsigned, BigInt> &bounds)
{
  std::map<unsigned, BigInt> by_stamp;
  forall_goto_functions (f, goto_functions)
    forall_goto_program_instructions (i, f->second.body)
    {
      const auto bound = bounds.find(i->loop_number);
      if (i->is_backwards_goto() && stamped_loop(*i) && bound != bounds.end())
        by_stamp[stamped_loop(*i)] = bound->second;
    }
  return by_stamp;
}
} // namespace

/// k-induction in which a satisfiable forward condition raises the bound of
/// each loop that hit its unwinding assertion to the iterations its guard
/// still allows, and in which loop invariants, guessed and proved on the
/// untransformed program, either prove it outright or strengthen the
/// inductive step.
///
/// Any per-loop bounds are sound for the base case and forward condition as
/// long as both use the same ones: together they then cover every execution.
/// The inductive step keeps the global k, which the base case at k covers.
int esbmc_parseoptionst::do_adaptive_kind_strategy(
  optionst &options,
  goto_functionst &goto_functions)
{
  const uint64_t max_k_step =
    cmdline.isset("unlimited-k-steps")
      ? std::numeric_limits<uint64_t>::max()
      : strtoul(cmdline.getval("max-k-step"), nullptr, 10);
  const uint64_t k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);
  const uint64_t k_step_base =
    strtoul(cmdline.getval("base-k-step"), nullptr, 10);
  const bool intervals = cmdline.isset("interval-analysis");

  const namespacet ns(context);
  stamp_loop_back_edges(goto_functions);
  const goto_functionst pristine = goto_functions;
  if (goto_k_induction(goto_functions, ns))
  {
    log_warning(
      "k-induction does not support loops that write array elements "
      "through a pointer yet. Disabling inductive step");
    options.set_option("disable-inductive-step", true);
  }
  if (intervals)
    instrument_loop_bounds_after_kind(goto_functions, ns, options);
  const std::map<unsigned, goto_programt::targett> placeholders =
    insert_kind_placeholders(goto_functions);

  // Cutting a loop deletes the interleaving points its body carried, so no
  // invariant the schema proves holds for a program with threads.
  const bool invariants_sound = !program_spawns_threads(pristine);
  std::vector<kind_candidatet> pool;
  if (invariants_sound)
  {
    goto_functionst scratch = pristine;
    pool = generate_kind_candidates(scratch, options, ns, intervals);
  }

  bmct::kind_feedbackt hints;
  hints.cap = BigInt(cmdline.getval("kind-max-bound"));
  std::map<unsigned, BigInt> bounds;
  std::set<unsigned> unbounded;
  std::vector<loop_head_samplet> samples;
  std::set<size_t> assumed, refuted;
  bool probe_pending = invariants_sound, raised_pending = false;
  fine_timet kind_time = 0, probe_time = 0, raised_time = 0;

  for (uint64_t k_step = k_step_base; k_step <= max_k_step;
       k_step += k_step_inc)
  {
    const fine_timet kind_start = current_time();
    // Schedules count steps, not k: --k-step 2 would otherwise never land a
    // power of two past the first step.
    const uint64_t step = (k_step - k_step_base) / k_step_inc + 1;

    // The raised bounds make a base case far costlier than one at k, and a
    // shallow bug needs none of them.
    options.set_option("unwindset", "");
    if (is_base_case_violated(options, goto_functions, k_step).is_true())
      return 1;

    // While a loop can outrun the cap the forward condition cannot hold, but
    // a loop left through a break may still stop early, so keep checking on a
    // doubling schedule rather than never.
    if (unbounded.empty() || is_power_of_two(step))
    {
      hints.remaining.clear();
      hints.samples.clear();
      const std::string raised = unwindset_of(bounds, k_step);
      options.set_option("unwindset", raised);
      options.set_option("adaptive-kind-defer-verdict", !raised.empty());
      const bool unwound =
        does_forward_condition_hold(options, goto_functions, k_step, &hints)
          .is_false();
      options.set_option("adaptive-kind-defer-verdict", false);
      if (unwound)
      {
        if (raised.empty())
          return 0;
        // No execution outruns the bounds, so a base case under exactly the
        // same ones covers every execution.
        if (is_base_case_violated(options, goto_functions, k_step).is_true())
          return 1;
        log_success("\nVERIFICATION SUCCESSFUL");
        return 0;
      }

      const bool jumped = apply_bound_hints(hints, k_step, bounds, unbounded);
      probe_pending |= jumped;
      raised_pending |= jumped;

      // Every state the counterexample reaches is one an invariant must hold
      // on, and the states together suggest relations worth guessing.
      if (invariants_sound && !hints.samples.empty())
      {
        samples.insert(
          samples.end(), hints.samples.begin(), hints.samples.end());
        probe_pending |= learn_from_samples(samples, assumed, pool, refuted);
      }
    }

    // A jump reaches executions a base case at k only gets to many steps
    // later. Search them once per jump, within the same budget as probing.
    fine_timet raised_now = 0;
    const std::string raised = unwindset_of(bounds, k_step);
    if (
      raised_pending && !raised.empty() &&
      raised_time <= std::max<fine_timet>(kind_time, 2000))
    {
      raised_pending = false;
      const fine_timet raised_start = current_time();
      options.set_option("unwindset", raised);
      if (is_base_case_violated(options, goto_functions, k_step).is_true())
        return 1;
      raised_now = current_time() - raised_start;
      raised_time += raised_now;
    }
    options.set_option("unwindset", "");

    if (k_step > 1)
    {
      bmct::kind_feedbackt counterexample;
      if (is_inductive_step_violated(
            options,
            goto_functions,
            k_step,
            invariants_sound ? &counterexample : nullptr)
            .is_false())
        return 0;
      // The state the inductive step started from is one a proven invariant
      // should rule out; bounds that hold where execution really goes but
      // not there are the guesses aimed at it.
      probe_pending |=
        learn_from_counterexample(samples, counterexample.samples, pool);
    }
    kind_time += current_time() - kind_start - raised_now;

    // Probing is a Houdini fixpoint over the whole program. Doing it on a
    // doubling schedule, and never for longer in total than k-induction
    // itself has run, keeps it from delaying a bug the base case is about to
    // find.
    const fine_timet budget = std::max<fine_timet>(kind_time, 2000);
    std::vector<size_t> live;
    for (size_t id = 0; id < pool.size(); ++id)
      if (!refuted.count(id))
        live.push_back(id);
    if (
      !probe_pending || live.empty() || step < 2 || !is_power_of_two(step) ||
      probe_time >= budget)
      continue;

    probe_pending = false;
    const fine_timet probe_start = current_time();
    const kind_proof_resultt proof = prove_kind_candidates(
      pristine,
      pool,
      live,
      probe_options(options, k_step),
      bounds_by_stamp(goto_functions, bounds),
      probe_start + budget - probe_time);
    probe_time += current_time() - probe_start;

    if (proof.program_proved)
    {
      log_success(
        "\nLoop invariants prove every property (k = {})\n"
        "VERIFICATION SUCCESSFUL",
        k_step);
      return 0;
    }

    for (size_t id : proof.proven)
    {
      const auto placeholder = placeholders.find(pool[id].loop);
      if (placeholder == placeholders.end() || !assumed.insert(id).second)
        continue;
      expr2tc &guard = placeholder->second->guard;
      guard = and2tc(guard, pool[id].expr);
      simplify(guard);
    }
    log_status(
      "{} of {} loop invariant candidate(s) proven, {} refuted by reachable "
      "states, in {} round(s) and {}s; assuming the proven ones in the "
      "inductive step",
      assumed.size(),
      pool.size(),
      refuted.size(),
      proof.rounds,
      time2string(current_time() - probe_start));
  }

  log_status("Unable to prove or falsify the program, giving up.");
  log_fail("VERIFICATION UNKNOWN");
  return 0;
}
