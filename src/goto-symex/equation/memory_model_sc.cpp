#include <goto-symex/equation/memory_model_sc.h>
#include <irep2/irep2_utils.h>

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
  constraints.clear();

  collect(steps);
  program_order();
  write_order();
  read_from();
  thread_order();
  atomic_sections();
  mutexes();

  steps.splice(steps.begin(), constraints);
  return true;
}

void memory_model_sct::collect(const symex_target_equationt::SSA_stepst &steps)
{
  for (const auto &step : steps)
  {
    if (!step.is_event() || step.ignore)
      continue;

    const size_t id = events.size();
    events.push_back(
      {&step,
       symbol2tc(get_uint32_type(), "__ESBMC_po_clock$" + std::to_string(id))});
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
        constrain(
          implies2tc(
            conjunction(
              {events[taken].step->guard,
               events[later].step->guard,
               before(taken, later)}),
            disjunction(releases)),
          events[later]);
      }
  }
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

bool memory_model_sct::same_thread(size_t a, size_t b) const
{
  return events[a].step->source.thread_nr == events[b].step->source.thread_nr;
}
