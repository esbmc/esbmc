#include <goto-symex/equation/memory_model_sc.h>
#include <irep2/irep2_utils.h>

#include <string>

namespace
{
irep_idt object_of(const symex_target_equationt::SSA_stept &step)
{
  return to_symbol2t(step.lhs).thename;
}
} // namespace

bool memory_model_sct::run(symex_target_equationt::SSA_stepst &steps)
{
  events.clear();
  per_thread.clear();
  reads.clear();
  writes.clear();
  constraints.clear();

  collect(steps);
  program_order();
  write_order();
  read_from();

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
      writes[object_of(step)].push_back(id);
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
        // A thread never reads its own later write.
        if (same_thread(w, r) && w > r)
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
               equality2tc(read.step->lhs, write.step->lhs),
               before(w, r)})),
          read);

        // From-read: a write ordered after the one read from comes after the
        // read too, so the read sees the latest write.
        for (size_t later : write_ids)
          if (later != w)
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
