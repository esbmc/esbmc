#include <goto-symex/equation/memory_model_sc.h>
#include <irep2/irep2_utils.h>

#include <algorithm>
#include <bit>
#include <set>
#include <string>
#include <vector>

namespace
{
irep_idt object_of(const symex_target_equationt::SSA_stept &step)
{
  return to_symbol2t(step.lhs).thename;
}

/// The value a read, or an update of part of an object, observes.
const expr2tc &value_read(const symex_target_equationt::SSA_stept &step)
{
  return step.type == symex_targett::SHARED_WRITE ? step.rhs : step.lhs;
}

/// \p a == \p b, element by element for an array of constant size: some
/// solvers, Bitwuzla among them, do not decide an equality with a constant
/// array that is not a top-level assertion.
expr2tc same_value(const expr2tc &a, const expr2tc &b)
{
  if (!is_array_type(a->type))
    return equality2tc(a, b);
  const array_type2t &type = to_array_type(a->type);
  if (type.size_is_infinite || !is_constant_int2t(type.array_size))
    return equality2tc(a, b);

  const uint64_t size = to_constant_int2t(type.array_size).value.to_uint64();
  std::vector<expr2tc> elems;
  for (uint64_t i = 0; i < size; ++i)
  {
    const expr2tc idx = constant_int2tc(index_type2(), BigInt(i));
    elems.push_back(same_value(
      index2tc(type.subtype, a, idx), index2tc(type.subtype, b, idx)));
  }
  return conjunction(elems);
}

unsigned thread_of(const symex_target_equationt::SSA_stept &step)
{
  return to_constant_int2t(step.lhs).value.to_uint64();
}
} // namespace

bool memory_model_sct::run(symex_target_equationt::SSA_stepst &steps)
{
  events.clear();
  per_thread.clear();
  reads.clear();
  writes.clear();
  spawns.clear();
  joins.clear();
  locks.clear();
  unlocks.clear();
  in_atomic.clear();
  constraints.clear();
  claims.clear();

  block_on_halted_joins(steps);
  collect(steps);
  program_order();
  write_order();
  read_from();
  thread_order();
  atomic_sections();
  mutexes();
  if (races)
    data_races();
  if (deadlocks)
  {
    complete_runs_only(steps);
    complete_runs_only(claims);
    claim_no_deadlock();
  }

  n_claims = claims.size();
  steps.splice(steps.begin(), constraints);
  steps.splice(steps.end(), claims);
  return true;
}

void memory_model_sct::collect(symex_target_equationt::SSA_stepst &steps)
{
  size_t n_events = 0;
  for (const auto &step : steps)
    n_events += step.is_event() && !step.ignore;
  // Wide enough for one clock value per event plus the deadlock check's stop
  // time above all of them.
  clock_type = get_uint_type(std::max<size_t>(1, std::bit_width(n_events)));

  for (auto &step : steps)
  {
    if (!step.is_event() || step.ignore)
      continue;

    const size_t id = events.size();
    step.cond = symbol2tc(clock_type, "__ESBMC_po_clock$" + std::to_string(id));
    events.push_back({&step, step.cond});
    per_thread[step.source.thread_nr].push_back(id);

    if (step.type == symex_targett::SHARED_READ)
      reads[object_of(step)].push_back(id);
    else if (step.type == symex_targett::SHARED_WRITE)
    {
      writes[object_of(step)].push_back(id);
      if (!is_nil_expr(step.rhs))
        reads[object_of(step)].push_back(id);
    }
    else if (step.type == symex_targett::SPAWN)
      spawns.push_back(id);
    else if (step.type == symex_targett::JOIN)
      joins.push_back(id);
    else if (step.type == symex_targett::LOCK)
      locks[object_of(step)].push_back(id);
    else if (step.type == symex_targett::UNLOCK)
      unlocks[object_of(step)].push_back(id);
  }
}

namespace
{
struct haltst
{
  std::map<unsigned, std::vector<expr2tc>> halts;
  std::map<unsigned, std::vector<std::pair<expr2tc, unsigned>>> joins;
  std::map<unsigned, expr2tc> stuck;

  /// When \p thread never ends: it halts at an unwinding bound, or it joins
  /// a thread that never ends.
  expr2tc never_ends(unsigned thread)
  {
    if (auto it = stuck.find(thread); it != stuck.end())
      return it->second;
    // A thread that joins itself, directly or not, waits at the join.
    stuck[thread] = gen_false_expr();
    std::vector<expr2tc> all = halts[thread];
    for (const auto &[guard, joined] : joins[thread])
      all.push_back(and2tc(guard, never_ends(joined)));
    return stuck[thread] = disjunction(all);
  }
};
} // namespace

/// A join on a thread that never ends does not return: the joining thread's
/// later events do not happen and its later claims hold.
void memory_model_sct::block_on_halted_joins(
  symex_target_equationt::SSA_stepst &steps) const
{
  haltst h;
  for (const auto &step : steps)
  {
    if (step.ignore)
      continue;
    if (step.type == symex_targett::HALT)
      h.halts[step.source.thread_nr].push_back(step.guard);
    else if (step.type == symex_targett::JOIN)
      h.joins[step.source.thread_nr].emplace_back(step.guard, thread_of(step));
  }
  if (h.halts.empty())
    return;

  std::map<unsigned, expr2tc> blocked;
  for (auto &step : steps)
  {
    if (step.ignore || !(step.is_event() || step.is_assert()))
      continue;
    expr2tc &now = blocked.try_emplace(step.source.thread_nr, gen_false_expr())
                     .first->second;
    if (step.is_assert())
    {
      step.cond = or2tc(now, step.cond);
      if (!is_nil_expr(step.cond_neg))
        step.cond_neg = and2tc(not2tc(now), step.cond_neg);
      continue;
    }
    // A join that never returns blocks from the join itself.
    if (step.type == symex_targett::JOIN)
      now = or2tc(now, and2tc(step.guard, h.never_ends(thread_of(step))));
    step.guard = and2tc(step.guard, not2tc(now));
  }
}

void memory_model_sct::program_order()
{
  for (const auto &[thread, ids] : per_thread)
    for (size_t i = 1; i < ids.size(); ++i)
      constrain(before(ids[i - 1], ids[i]), events[ids[i]]);
}

void memory_model_sct::write_order()
{
  for (const auto &[object, ids] : writes)
    for (size_t i = 0; i < ids.size(); ++i)
      for (size_t j = i + 1; j < ids.size(); ++j)
        if (!same_thread(ids[i], ids[j]))
          constrain(
            notequal2tc(events[ids[i]].clock, events[ids[j]].clock),
            events[ids[j]]);
}

void memory_model_sct::read_from()
{
  for (const auto &[object, read_ids] : reads)
  {
    auto it = writes.find(object);
    // With no write to read from, the read keeps its unconstrained value.
    if (it == writes.end())
      continue;
    const std::vector<size_t> &write_ids = it->second;

    for (size_t r : read_ids)
    {
      const eventt &read = events[r];
      std::vector<expr2tc> choices;
      for (size_t w : write_ids)
      {
        // A thread never reads its own later write, nor an update its own.
        if (same_thread(w, r) && w >= r)
          continue;

        const eventt &write = events[w];
        expr2tc rf = symbol2tc(
          get_bool_type(),
          "__ESBMC_po_rf$" + std::to_string(w) + "$" + std::to_string(r));
        choices.push_back(rf);

        constrain(
          implies2tc(
            rf,
            conjunction(
              {write.step->guard,
               same_value(value_read(*read.step), write.step->lhs),
               before(w, r)})),
          read);

        // From-read: a write ordered after the one read from comes after the
        // read too, so the read sees the latest write. For an update this
        // also keeps every other write out from between the two.
        for (size_t later : write_ids)
          if (later != w && later != r)
            constrain(
              implies2tc(
                conjunction({rf, events[later].step->guard, before(w, later)}),
                before(r, later)),
              read);
      }

      if (!choices.empty())
        constrain(implies2tc(read.step->guard, disjunction(choices)), read);
    }
  }
}

void memory_model_sct::thread_order()
{
  for (size_t s : spawns)
  {
    auto it = per_thread.find(thread_of(*events[s].step));
    if (it != per_thread.end())
      constrain(
        implies2tc(events[s].step->guard, before(s, it->second.front())),
        events[s]);
  }

  for (size_t j : joins)
  {
    auto it = per_thread.find(thread_of(*events[j].step));
    if (it != per_thread.end())
      constrain(
        implies2tc(events[j].step->guard, before(it->second.back(), j)),
        events[j]);
  }
}

void memory_model_sct::atomic_sections()
{
  for (const auto &[thread, ids] : per_thread)
  {
    unsigned depth = 0;
    size_t begin = 0;
    for (size_t id : ids)
    {
      const symex_targett::step_typet type = events[id].step->type;
      if (type == symex_targett::ATOMIC_BEGIN && depth++ == 0)
        begin = id;
      else if (type == symex_targett::ATOMIC_END && depth > 0 && --depth == 0)
        isolate(begin, id);
      else if (type == symex_targett::HALT && depth > 0)
        halt_inside(begin, id);
      else if (depth > 0)
        in_atomic.insert(id);
    }
    // A section still open when its thread ends lasts to the thread's end.
    if (depth > 0)
      isolate(begin, ids.back());
  }
}

void memory_model_sct::isolate(size_t begin, size_t end)
{
  for (size_t other = 0; other < events.size(); ++other)
    if (!same_thread(other, begin))
      constrain(
        implies2tc(
          events[begin].step->guard,
          or2tc(before(other, begin), before(end, other))),
        events[begin]);
}

/// A thread that halts inside an atomic section never leaves it, so every
/// other event comes before the section.
void memory_model_sct::halt_inside(size_t begin, size_t halt)
{
  for (size_t other = 0; other < events.size(); ++other)
    if (!same_thread(other, begin))
      constrain(
        implies2tc(
          conjunction({events[halt].step->guard, events[other].step->guard}),
          before(other, begin)),
        events[halt]);
}

void memory_model_sct::mutexes()
{
  for (const auto &[mutex, lock_ids] : locks)
  {
    const std::vector<size_t> &unlock_ids = unlocks[mutex];
    for (size_t taken : lock_ids)
      for (size_t later : lock_ids)
      {
        if (taken == later)
          continue;
        if (taken < later && !same_thread(taken, later))
          constrain(
            notequal2tc(events[taken].clock, events[later].clock),
            events[later]);

        // The false seed: disjunction() of nothing is true.
        std::vector<expr2tc> releases{gen_false_expr()};
        for (size_t u : unlock_ids)
          releases.push_back(conjunction(
            {events[u].step->guard, before(taken, u), before(u, later)}));
        std::vector<expr2tc> both{
          events[taken].step->guard,
          events[later].step->guard,
          before(taken, later)};
        if (deadlocks)
          both.push_back(done(later));
        constrain(
          implies2tc(conjunction(both), disjunction(releases)), events[later]);
      }
  }
}

void memory_model_sct::data_races()
{
  for (const auto &[object, write_ids] : writes)
  {
    std::set<size_t> accesses(write_ids.begin(), write_ids.end());
    if (auto it = reads.find(object); it != reads.end())
      accesses.insert(it->second.begin(), it->second.end());
    for (size_t w : write_ids)
      for (size_t other : accesses)
        if (
          !same_thread(w, other) &&
          !(in_atomic.count(w) && in_atomic.count(other)) &&
          (w < other ||
           events[other].step->type != symex_targett::SHARED_WRITE))
          claim_no_race(w, other);
  }
}

/// A race is a pair of conflicting events with no event between them in some
/// execution: the schedule can then run either one first. An atomic section's
/// bounds access no memory, so they do not separate a plain access from one
/// inside the section (#2386).
void memory_model_sct::claim_no_race(size_t a, size_t b)
{
  std::vector<expr2tc> adjacent{or2tc(before(a, b), before(b, a))};
  for (size_t c = 0; c < events.size(); ++c)
    if (c != a && c != b && !is_atomic_bound(c))
      adjacent.push_back(implies2tc(
        events[c].step->guard,
        not2tc(or2tc(
          and2tc(before(a, c), before(c, b)),
          and2tc(before(b, c), before(c, a))))));

  const eventt &second = events[b];
  const bool both_write = second.step->type == symex_targett::SHARED_WRITE;
  claims.emplace_back();
  symex_target_equationt::SSA_stept &step = claims.back();
  step.type = symex_targett::ASSERT;
  step.guard = and2tc(events[a].step->guard, second.step->guard);
  // The encoding reads only cond, as for a claim symex makes.
  step.cond_neg = and2tc(step.guard, conjunction(adjacent));
  step.cond = not2tc(step.cond_neg);
  step.source = second.step->source;
  std::string comment = both_write ? "W/W" : "R/W";
  comment += " data race on ";
  comment += id2string(object_of(*second.step));
  step.comment = comment;
}

/// A deadlock is a prefix after which main, and every other thread that has
/// events left, waits at its first pending event. Returning from main ends
/// the process, so main must be one of the waiting threads.
void memory_model_sct::claim_no_deadlock()
{
  auto main_it = per_thread.find(0);
  if (main_it == per_thread.end())
    return;

  std::vector<expr2tc> all_wait;
  for (const auto &[thread, ids] : per_thread)
  {
    std::vector<expr2tc> waiting{gen_false_expr()};
    // Shared across the iterations, so the formula grows linearly.
    expr2tc earlier_done = gen_true_expr();
    for (size_t id : ids)
    {
      const expr2tc &guard = events[id].step->guard;
      waiting.push_back(
        conjunction({guard, not2tc(done(id)), earlier_done, waits(id)}));
      earlier_done = and2tc(earlier_done, implies2tc(guard, done(id)));
    }
    all_wait.push_back(
      thread == 0 ? disjunction(waiting)
                  : or2tc(finished(thread), disjunction(waiting)));
  }

  claims.emplace_back();
  symex_target_equationt::SSA_stept &step = claims.back();
  step.type = symex_targett::ASSERT;
  step.guard = gen_true_expr();
  step.cond_neg = conjunction(all_wait);
  step.cond = not2tc(step.cond_neg);
  step.source = events[main_it->second.back()].step->source;
  step.comment = "Deadlocked state";
}

/// Whether \p id, pending at the stop time, cannot run: it locks a mutex
/// whose last lock in the prefix has no release after it, or joins a thread
/// with events left.
expr2tc memory_model_sct::waits(size_t id) const
{
  const symex_target_equationt::SSA_stept &step = *events[id].step;
  if (step.type == symex_targett::JOIN)
    return not2tc(finished(thread_of(step)));
  if (step.type != symex_targett::LOCK)
    return gen_false_expr();

  auto unlock_it = unlocks.find(object_of(step));
  std::vector<expr2tc> held{gen_false_expr()};
  for (size_t taken : locks.at(object_of(step)))
  {
    std::vector<expr2tc> released{gen_false_expr()};
    if (unlock_it != unlocks.end())
      for (size_t u : unlock_it->second)
        released.push_back(
          conjunction({events[u].step->guard, done(u), before(taken, u)}));
    held.push_back(conjunction(
      {events[taken].step->guard, done(taken), not2tc(disjunction(released))}));
  }
  return disjunction(held);
}

/// Each claim other than the deadlock one is violated only by an execution
/// that runs every thread to its end, as without a stop time.
void memory_model_sct::complete_runs_only(
  symex_target_equationt::SSA_stepst &steps) const
{
  std::vector<expr2tc> all;
  for (const auto &[thread, ids] : per_thread)
    all.push_back(finished(thread));
  const expr2tc all_finished = conjunction(all);

  for (auto &step : steps)
    if (step.is_assert())
    {
      step.cond = implies2tc(all_finished, step.cond);
      if (!is_nil_expr(step.cond_neg))
        step.cond_neg = and2tc(all_finished, step.cond_neg);
    }
}

expr2tc memory_model_sct::finished(unsigned thread) const
{
  std::vector<expr2tc> all;
  if (auto it = per_thread.find(thread); it != per_thread.end())
    for (size_t id : it->second)
      all.push_back(implies2tc(events[id].step->guard, done(id)));
  return conjunction(all);
}

expr2tc memory_model_sct::done(size_t id) const
{
  return lessthan2tc(
    events[id].clock, symbol2tc(clock_type, "__ESBMC_po_stop"));
}

void memory_model_sct::constrain(const expr2tc &cond, const eventt &about)
{
  constraints.emplace_back();
  symex_target_equationt::SSA_stept &step = constraints.back();
  step.type = symex_targett::ASSUME;
  step.guard = gen_true_expr();
  step.cond = cond;
  step.source = about.step->source;
  step.hidden = true;
}

expr2tc memory_model_sct::before(size_t a, size_t b) const
{
  return lessthan2tc(events[a].clock, events[b].clock);
}

bool memory_model_sct::is_atomic_bound(size_t id) const
{
  const symex_targett::step_typet type = events[id].step->type;
  return type == symex_targett::ATOMIC_BEGIN ||
         type == symex_targett::ATOMIC_END;
}

bool memory_model_sct::same_thread(size_t a, size_t b) const
{
  return events[a].step->source.thread_nr == events[b].step->source.thread_nr;
}
