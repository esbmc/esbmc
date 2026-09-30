#include <transition-system/ts_slice.h>

#include <irep2/irep2_utils.h>

#include <algorithm>

void slice_transition_system(transition_systemt &ts)
{
  std::unordered_map<expr2tc, expr2tc, irep2_hash> def_of;
  for (const auto *defs : {&ts.prefix_defs, &ts.body_defs})
    for (const auto &e : *defs)
      def_of.emplace(to_equality2t(e).side_1, to_equality2t(e).side_2);
  std::unordered_map<expr2tc, unsigned, irep2_hash> state_of;
  for (unsigned k = 0; k < ts.states.size(); k++)
    state_of.emplace(ts.states[k].pre, k);

  std::vector<bool> live(ts.states.size(), false);
  std::unordered_set<expr2tc, irep2_hash> needed;
  std::unordered_set<const expr2t *> visited;
  std::vector<expr2tc> work;
  auto need = [&](const expr2tc &e)
  {
    std::unordered_set<expr2tc, irep2_hash> syms;
    collect_symbols(e, syms, visited);
    for (const auto &s : syms)
      if (needed.insert(s).second)
        work.push_back(s);
  };

  for (const auto *roots :
       {&ts.body_assumes, &ts.invariants, &ts.prefix_assumes})
    for (const auto &e : *roots)
      need(e);
  for (const auto *props : {&ts.bad, &ts.prefix_bad})
    for (const auto &p : *props)
      need(p.violated);
  need(ts.back_guard);

  while (!work.empty())
  {
    expr2tc s = work.back();
    work.pop_back();
    auto d = def_of.find(s);
    if (d != def_of.end())
      need(d->second);
    auto st = state_of.find(s);
    if (st != state_of.end())
    {
      live[st->second] = true;
      need(ts.states[st->second].post);
      need(ts.states[st->second].init);
    }
  }

  auto keep_defs = [&](std::vector<expr2tc> &defs)
  {
    defs.erase(
      std::remove_if(
        defs.begin(),
        defs.end(),
        [&](const expr2tc &e)
        { return !needed.count(to_equality2t(e).side_1); }),
      defs.end());
  };
  keep_defs(ts.prefix_defs);
  keep_defs(ts.body_defs);

  std::vector<transition_systemt::statet> kept;
  for (unsigned k = 0; k < ts.states.size(); k++)
    if (live[k])
      kept.push_back(ts.states[k]);
  ts.states = std::move(kept);

  ts.inputs.erase(
    std::remove_if(
      ts.inputs.begin(),
      ts.inputs.end(),
      [&](const std::pair<expr2tc, std::string> &in)
      { return !needed.count(in.first); }),
    ts.inputs.end());
}
