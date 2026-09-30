#include <transition-system/transition_system.h>

#include <irep2/irep2_utils.h>

#include <langapi/language_util.h>

#include <algorithm>
#include <functional>
#include <ostream>

void collect_symbols(
  const expr2tc &e,
  std::unordered_set<expr2tc, irep2_hash> &out,
  std::unordered_set<const expr2t *> &seen)
{
  if (is_nil_expr(e) || !seen.insert(e.get()).second)
    return;
  if (is_symbol2t(e))
  {
    out.insert(e);
    return;
  }
  e->foreach_operand([&](const expr2tc &op)
                     { collect_symbols(op, out, seen); });
}

bool transition_systemt::has_local(const expr2tc &e) const
{
  if (is_nil_expr(e))
    return false;
  if (is_symbol2t(e))
    return step_local.count(e) != 0;
  auto it = has_local_memo.find(e.get());
  if (it != has_local_memo.end())
    return it->second;
  bool found = false;
  e->foreach_operand([&](const expr2tc &op) { found |= has_local(op); });
  has_local_memo.emplace(e.get(), found);
  return found;
}

void transition_systemt::set_step_local(step_localt syms)
{
  step_local = std::move(syms);
  has_local_memo.clear();
}

expr2tc transition_systemt::at_step(const expr2tc &e, unsigned step) const
{
  return ts_step_renamert(*this, step)(e);
}

ts_step_renamert::ts_step_renamert(const transition_systemt &ts, unsigned step)
  : ts(ts), suffix("$ts" + std::to_string(step))
{
}

expr2tc ts_step_renamert::operator()(const expr2tc &x)
{
  if (!ts.has_local(x))
    return x;
  auto it = memo.find(x.get());
  if (it != memo.end())
    return it->second;
  expr2tc out;
  if (is_symbol2t(x))
    out = symbol2tc(x->type, to_symbol2t(x).get_symbol_name() + suffix);
  else
  {
    out = x;
    out.get()->Foreach_operand(
      [&](expr2tc &op)
      {
        if (!is_nil_expr(op))
          op = (*this)(op);
      });
  }
  memo.emplace(x.get(), out);
  return out;
}

void transition_systemt::dump(std::ostream &out, const namespacet &ns) const
{
  // Printing expands the DAG into a tree, which can be exponentially larger.
  const size_t limit = 400;
  std::unordered_map<const expr2t *, size_t> sizes;
  std::function<size_t(const expr2tc &)> tree_size =
    [&](const expr2tc &e) -> size_t
  {
    if (is_nil_expr(e))
      return 0;
    auto it = sizes.find(e.get());
    if (it != sizes.end())
      return it->second;
    size_t n = 1;
    e->foreach_operand([&](const expr2tc &op)
                       { n = std::min(limit + 1, n + tree_size(op)); });
    sizes.emplace(e.get(), n);
    return n;
  };
  auto show = [&](const expr2tc &e)
  {
    return tree_size(e) > limit ? std::string("<large expression>")
                                : from_expr(ns, "", e);
  };
  out << "Transition system: " << states.size() << " state variables, "
      << inputs.size() << " inputs, " << body_defs.size()
      << " step definitions, " << body_assumes.size() << " step assumptions, "
      << bad.size() << " properties, " << invariants.size()
      << " invariants; prefix: " << prefix_defs.size() << " definitions, "
      << prefix_assumes.size() << " assumptions, " << prefix_bad.size()
      << " properties\n";
  out << "State (name: init | pre | post):\n";
  for (const auto &s : states)
    out << "  " << s.name << ": " << show(s.init) << " | " << show(s.pre)
        << " | " << show(s.post) << "\n";
  out << "Inputs:\n";
  for (const auto &[sym, label] : inputs)
    out << "  " << show(sym) << "  (" << label << ")\n";
  out << "Invariants:\n";
  for (const auto &e : invariants)
    out << "  " << show(e) << "\n";
  out << "Prefix assumptions:\n";
  for (const auto &e : prefix_assumes)
    out << "  " << show(e) << "\n";
  out << "Step definitions:\n";
  for (const auto &e : body_defs)
    out << "  " << show(e) << "\n";
  out << "Step assumptions:\n";
  for (const auto &e : body_assumes)
    out << "  " << show(e) << "\n";
  out << "Back-edge guard: " << show(back_guard) << "\n";
  out << "Bad:\n";
  for (const auto &p : bad)
    out << "  [" << p.location.as_string() << "] " << p.comment << ": "
        << show(p.violated) << "\n";
}
