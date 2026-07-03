#include <goto-programs/dead_store_analysis.h>

#include <charconv>
#include <goto-programs/goto_functions.h>
#include <map>
#include <set>
#include <unordered_map>
#include <util/prefix.h>

namespace
{
/// Scalar types whose stores we reason about. Aggregates and code are out of
/// scope; a dead store to one of those needs finer-grained (per-field /
/// per-element) liveness we deliberately do not attempt here.
bool is_tracked_scalar(const type2tc &t)
{
  return is_bool_type(t) || is_signedbv_type(t) || is_unsignedbv_type(t) ||
         is_fixedbv_type(t) || is_floatbv_type(t) || is_pointer_type(t);
}

/// Walk an l-value chain and return its root symbol name, or empty if the
/// chain bottoms out in something other than a symbol.
irep_idt lvalue_root(const expr2tc &e)
{
  const expr2tc *cur = &e;
  while (true)
  {
    if (is_nil_expr(*cur))
      return irep_idt();
    if (is_symbol2t(*cur))
      return to_symbol2t(*cur).thename;
    if (is_index2t(*cur))
      cur = &to_index2t(*cur).source_value;
    else if (is_member2t(*cur))
      cur = &to_member2t(*cur).source_value;
    else if (is_dereference2t(*cur))
      cur = &to_dereference2t(*cur).value;
    else
      return irep_idt();
  }
}

/// Collect the names of every symbol read within `expr` (all symbol leaves).
void collect_symbols(const expr2tc &expr, std::vector<irep_idt> &out)
{
  if (is_nil_expr(expr))
    return;
  if (is_symbol2t(expr))
    out.push_back(to_symbol2t(expr).thename);
  expr->foreach_operand(
    [&out](const expr2tc &op) { collect_symbols(op, out); });
}

/// Record the root symbol of every `address_of` target reachable in `expr`.
/// Such variables may be read through the resulting pointer, so they are
/// excluded from tracking to keep the analysis sound without alias analysis.
void collect_address_taken(const expr2tc &expr, std::set<irep_idt> &out)
{
  if (is_nil_expr(expr))
    return;
  if (is_address_of2t(expr))
  {
    irep_idt root = lvalue_root(to_address_of2t(expr).ptr_obj);
    if (!root.empty())
      out.insert(root);
  }
  expr->foreach_operand(
    [&out](const expr2tc &op) { collect_address_taken(op, out); });
}

/// A dead store never originates in a system header or a compiler-synthesised
/// location; restricting to real source keeps advisories on user code and off
/// the operational-model library baked into the goto program. The pass runs
/// before inlining, so every linked OM/library function with a body is still
/// present — those live under the extracted-headers temp dir ("-headers-") at
/// runtime and under c2goto/library/ for in-tree builds (mirrors
/// remove_exceptions.cpp::is_library_function).
bool is_reportable_location(const locationt &loc)
{
  const std::string file = loc.get_file().as_string();
  if (file.empty() || file == "<built-in>" || file == "<builtin>")
    return false;
  if (has_prefix(file, "/usr/"))
    return false;
  return file.find("-headers-") == std::string::npos &&
         file.find("c2goto/library/") == std::string::npos;
}

/// Non-throwing decimal parse of a source line number; 0 on empty/non-numeric
/// input (a nil location). Matches the parse used by the SARIF writer.
unsigned parse_line(const std::string &s)
{
  unsigned v = 0;
  std::from_chars(s.data(), s.data() + s.size(), v);
  return v;
}
} // namespace

bool goto_check_dead_store::runOnFunction(
  std::pair<const irep_idt, goto_functiont> &F)
{
  if (!F.second.body_available)
    return false;

  goto_programt &body = F.second.body;
  const auto end = body.instructions.cend();

  // Index every instruction so CFG successors resolve to positions.
  std::vector<goto_programt::const_targett> insns;
  std::unordered_map<const goto_programt::instructiont *, size_t> pos;
  for (auto it = body.instructions.cbegin(); it != end; ++it)
  {
    pos.emplace(&*it, insns.size());
    insns.push_back(it);
  }
  const size_t n = insns.size();
  if (n == 0)
    return false;

  // Pass 1: candidate scalar locals from DECLs, minus any address-taken var.
  std::set<irep_idt> address_taken;
  std::set<irep_idt> candidates;
  for (const auto &it : insns)
  {
    collect_address_taken(it->code, address_taken);
    collect_address_taken(it->guard, address_taken);

    if (!it->is_decl())
      continue;
    const irep_idt &name = to_code_decl2t(it->code).value;
    const symbolt *s = context.find_symbol(name);
    if (!s || s->static_lifetime || s->is_extern || !s->lvalue)
      continue;
    if (has_prefix(s->name, "return_value$"))
      continue;
    if (has_prefix(s->id, "__ESBMC_") || has_prefix(s->name, "__ESBMC_"))
      continue;
    // A write to a volatile object is an observable side effect (C11
    // §5.1.2.3), never a dead store. The `#volatile` qualifier survives only on
    // the legacy type — get_type2()/migrate_type drops it.
    if (s->get_type().cmt_volatile())
      continue;
    if (is_tracked_scalar(s->get_type2()))
      candidates.insert(name);
  }

  std::map<irep_idt, size_t> track; // tracked name -> bit index
  for (const irep_idt &c : candidates)
    if (!address_taken.count(c))
      track.emplace(c, track.size());
  const size_t bits = track.size();
  if (bits == 0)
    return false;

  auto bit_of = [&track](const irep_idt &name) -> long {
    auto it = track.find(name);
    return it == track.end() ? -1 : static_cast<long>(it->second);
  };

  // Pass 2: per-instruction use/def sets over the tracked universe, plus the
  // list of plain-assignment sites that are candidate dead stores.
  std::vector<std::vector<bool>> use(n, std::vector<bool>(bits, false));
  std::vector<std::vector<bool>> def(n, std::vector<bool>(bits, false));
  std::vector<std::vector<size_t>> succ(n);

  struct sitet
  {
    size_t idx;
    size_t bit;
    irep_idt name;
  };
  std::vector<sitet> sites;

  for (size_t i = 0; i < n; ++i)
  {
    const auto &it = insns[i];

    goto_programt::const_targetst succs;
    body.get_successors(it, succs);
    for (const auto &s : succs)
      if (s != end)
        succ[i].push_back(pos.at(&*s));

    std::vector<irep_idt> reads;
    long def_bit = -1;

    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      collect_symbols(a.source, reads);
      if (is_symbol2t(a.target))
      {
        def_bit = bit_of(to_symbol2t(a.target).thename);
        if (def_bit >= 0)
          sites.push_back(
            {i, static_cast<size_t>(def_bit), to_symbol2t(a.target).thename});
      }
      else
        collect_symbols(a.target, reads);
    }
    else if (it->is_function_call())
    {
      const code_function_call2t &fc = to_code_function_call2t(it->code);
      collect_symbols(fc.function, reads);
      for (const auto &arg : fc.operands)
        collect_symbols(arg, reads);
      if (is_symbol2t(fc.ret))
        def_bit = bit_of(to_symbol2t(fc.ret).thename);
      else
        collect_symbols(fc.ret, reads);
    }
    else if (it->is_decl())
    {
      def_bit = bit_of(to_code_decl2t(it->code).value);
    }
    else
    {
      collect_symbols(it->code, reads);
      collect_symbols(it->guard, reads);
    }

    for (const irep_idt &r : reads)
    {
      long b = bit_of(r);
      if (b >= 0)
        use[i][b] = true;
    }
    if (def_bit >= 0)
      def[i][def_bit] = true;
  }

  // Backward live-variable fixpoint:
  //   live_out(i) = U live_in(s) over successors s
  //   live_in(i)  = use(i) U (live_out(i) \ def(i))
  std::vector<std::vector<bool>> live_in(n, std::vector<bool>(bits, false));
  std::vector<std::vector<bool>> live_out(n, std::vector<bool>(bits, false));
  bool changed = true;
  while (changed)
  {
    changed = false;
    for (size_t i = n; i-- > 0;)
    {
      for (size_t b = 0; b < bits; ++b)
      {
        bool out_b = false;
        for (size_t s : succ[i])
          if (live_in[s][b])
          {
            out_b = true;
            break;
          }
        if (out_b != live_out[i][b])
        {
          live_out[i][b] = out_b;
          changed = true;
        }
        bool in_b = use[i][b] || (live_out[i][b] && !def[i][b]);
        if (in_b != live_in[i][b])
        {
          live_in[i][b] = in_b;
          changed = true;
        }
      }
    }
  }

  // A plain assignment to a tracked local is a dead store when its written
  // value is not live on exit.
  for (const sitet &site : sites)
  {
    if (live_out[site.idx][site.bit])
      continue;
    const locationt &loc = insns[site.idx]->location;
    if (!is_reportable_location(loc))
      continue;

    const symbolt *s = context.find_symbol(site.name);
    const std::string name = s ? id2string(s->name) : id2string(site.name);

    dead_store_advisoryt adv;
    adv.lhs_name = name;
    adv.comment = "dead store: assignment to " + name + " never read";
    adv.file = loc.get_file().as_string();
    adv.function = loc.get_function().as_string();
    adv.line = parse_line(loc.get_line().as_string());
    advisories.push_back(std::move(adv));
  }

  return false;
}
