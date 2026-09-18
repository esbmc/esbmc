#include <esbmc/ts/ts_pdr.h>

#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/base/time_stopping.h>
#include <util/lang/c_types.h>
#include <util/message/message.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace
{
/// Literal-dropping attempts per generalisation, after the unsat core, and
/// the largest cube worth trying them on.
const unsigned drop_budget = 16;
const size_t drop_max_cube = 128;
} // namespace

ts_pdrt::ts_pdrt(
  const transition_systemt &ts,
  const namespacet &ns,
  optionst &options)
  : ts_enginet(ts, ns, options, false, true, true), options_(&options)
{
  add_step(0);
  act_trans = symbol2tc(get_bool_type(), "ts$act$trans");
  std::vector<expr2tc> trans;
  ts_step_renamert at(ts, 0);
  for (const expr2tc &a : ts.body_assumes)
    trans.push_back(at(a));
  trans.push_back(at(ts.back_guard));
  std::vector<expr2tc> violated;
  for (const auto &p : ts.bad)
    violated.push_back(at(p.violated));
  bad_expr = violated.empty() ? gen_false_expr() : disjunction(violated);
  trans.push_back(not2tc(bad_expr));
  trans_expr = conjunction(trans);
  solver->assert_expr(implies2tc(act_trans, trans_expr));
}

namespace
{
void collect_shared(
  const expr2tc &e,
  const transition_systemt &ts,
  std::map<std::string, expr2tc> &out,
  std::unordered_set<const expr2t *> &seen)
{
  if (is_nil_expr(e) || !seen.insert(e.get()).second)
    return;
  if (is_symbol2t(e))
  {
    if (!ts.step_local.count(e))
      out.emplace(to_symbol2t(e).get_symbol_name(), e);
    return;
  }
  e->foreach_operand(
    [&](const expr2tc &op) { collect_shared(op, ts, out, seen); });
}
} // namespace

bool ts_pdrt::build_bits()
{
  // State components: the loop's live state, then every value fixed before
  // the loop that the step reads. The latter never change, but a cube that
  // omitted them would let each query choose them afresh.
  std::vector<std::pair<expr2tc, expr2tc>> components;
  std::vector<std::string> names;
  for (unsigned k = 0; k < ts.state_pre.size(); k++)
  {
    components.emplace_back(
      ts.at_step(ts.state_pre[k], 0), ts.at_step(ts.state_post[k], 0));
    names.push_back(ts.state_names[k]);
  }
  std::map<std::string, expr2tc> shared;
  std::unordered_set<const expr2t *> seen;
  for (const auto *es : {&ts.body_defs, &ts.body_assumes, &ts.invariants})
    for (const expr2tc &e : *es)
      collect_shared(e, ts, shared, seen);
  for (const auto &p : ts.bad)
    collect_shared(p.violated, ts, shared, seen);
  for (const expr2tc &e : ts.state_post)
    collect_shared(e, ts, shared, seen);
  collect_shared(ts.back_guard, ts, shared, seen);
  for (const auto &[name, sym] : shared)
  {
    components.emplace_back(sym, sym);
    names.push_back(name);
  }

  const type2tc bit_type = get_uint_type(1);
  const expr2tc one = from_integer(1, bit_type);
  // Cube literals are assumed on every query. Bitwuzla preprocesses any
  // assumption that is not a plain Boolean symbol, so name each bit once.
  auto as_symbol = [&](const expr2tc &e, const char *side) {
    if (is_symbol2t(e))
      return e;
    expr2tc s = symbol2tc(
      get_bool_type(),
      "ts$pdr$bit$" + std::to_string(cur_bit.size()) + "$" + side);
    solver->assert_expr(equality2tc(s, e));
    return s;
  };
  auto add_bit = [&](const expr2tc &c, const expr2tc &n, bool shared) {
    expr2tc cs = as_symbol(c, "cur");
    next_bit.push_back(shared ? cs : as_symbol(n, "next"));
    cur_bit.push_back(cs);
  };
  for (unsigned k = 0; k < components.size(); k++)
  {
    const auto &[cur, next] = components[k];
    component_cur.push_back(cur);
    const type2tc &t = cur->type;
    expr2tc init = k < ts.state_init.size() ? ts.state_init[k] : expr2tc();
    auto init_value = [&](unsigned b) {
      if (!is_nil_expr(init) && is_constant_bool2t(init))
        return to_constant_bool2t(init).value ? 1 : 0;
      if (!is_nil_expr(init) && is_constant_int2t(init))
        return (to_constant_int2t(init).value.to_uint64() >> b) & 1 ? 1 : 0;
      return -1;
    };
    const bool shared = cur == next;
    if (is_bool_type(t))
    {
      bit_of.emplace_back(k, 0);
      add_bit(cur, next, shared);
      init_bit.push_back(init_value(0));
    }
    else if (is_bv_type(t) && t->get_width() <= 64)
      for (unsigned b = 0; b < t->get_width(); b++)
      {
        bit_of.emplace_back(k, b);
        add_bit(
          equality2tc(extract2tc(bit_type, cur, b, b), one),
          equality2tc(extract2tc(bit_type, next, b, b), one),
          shared);
        init_bit.push_back(init_value(b));
      }
    else
    {
      reason = "state component " + names[k] +
               " is not a bit-vector of at most 64 bits";
      return false;
    }
  }
  return true;
}

expr2tc ts_pdrt::literal(const litt &l, bool next) const
{
  const expr2tc &e = next ? next_bit[l.bit] : cur_bit[l.bit];
  return l.value ? e : not2tc(e);
}

expr2tc ts_pdrt::cube_expr(const cubet &c, bool next) const
{
  std::vector<expr2tc> lits;
  for (const litt &l : c)
    lits.push_back(literal(l, next));
  return conjunction(lits);
}

std::vector<expr2tc> ts_pdrt::frame(unsigned level) const
{
  if (level == 0)
    return {act_init, act_inv};
  std::vector<expr2tc> lits{act_inv};
  for (unsigned j = level; j < level_lits.size(); j++)
    lits.push_back(level_lits[j]);
  return lits;
}

smt_resultt ts_pdrt::solve(
  const std::vector<expr2tc> &assumptions,
  const std::string &kind)
{
  queries++;
  fine_timet start = current_time();
  smt_resultt res = solver->dec_solve_assuming(assumptions);
  auto &[count, seconds] = query_stats[kind];
  count++;
  seconds += std::stod(time2string(current_time() - start));
  if (res != smt_resultt::P_SATISFIABLE && res != smt_resultt::P_UNSATISFIABLE)
    throw std::runtime_error("solver error");
  return res;
}

ts_pdrt::cubet ts_pdrt::model_cube()
{
  std::vector<BigInt> values;
  for (const expr2tc &component : component_cur)
  {
    expr2tc v = solver->get(component);
    if (is_constant_bool2t(v))
      values.push_back(to_constant_bool2t(v).value ? 1 : 0);
    else if (is_constant_int2t(v))
      values.push_back(to_constant_int2t(v).value);
    else
      throw std::runtime_error("state value missing from the model");
  }
  cubet c;
  for (unsigned i = 0; i < bit_of.size(); i++)
  {
    const auto [k, b] = bit_of[i];
    c.push_back({i, ((values[k].to_uint64() >> b) & 1) != 0});
  }
  return c;
}

ts_pdrt::cubet ts_pdrt::lift(const cubet &s, const expr2tc &target)
{
  std::vector<expr2tc> inputs;
  for (const auto &input : ts.inputs)
  {
    expr2tc sym = ts.at_step(input.first, 0);
    expr2tc value = solver->get(sym);
    if (!is_nil_expr(value) && is_constant_expr(value))
      inputs.push_back(equality2tc(sym, value));
  }
  std::vector<expr2tc> assumptions{conjunction(inputs), not2tc(target)};
  std::unordered_map<expr2tc, unsigned, irep2_hash> index;
  for (unsigned i = 0; i < s.size(); i++)
  {
    expr2tc lit = literal(s[i], false);
    index.emplace(lit, i);
    assumptions.push_back(lit);
  }
  if (solve(assumptions, "lift") == smt_resultt::P_SATISFIABLE)
    return s;
  cubet core;
  for (const expr2tc &u : solver->unsat_assumptions())
  {
    auto it = index.find(u);
    if (it != index.end())
      core.push_back(s[it->second]);
  }
  std::sort(core.begin(), core.end());
  return core;
}

bool ts_pdrt::intersects_init(const cubet &c)
{
  bool symbolic = false;
  for (const litt &l : c)
  {
    if (init_bit[l.bit] < 0)
      symbolic = true;
    else if (init_bit[l.bit] != (l.value ? 1 : 0))
      return false;
  }
  if (!symbolic)
    return true;
  return solve({act_init, act_inv, cube_expr(c, false)}, "init") ==
         smt_resultt::P_SATISFIABLE;
}

bool ts_pdrt::subsumed(const cubet &c, unsigned level) const
{
  for (unsigned j = level; j < frames.size(); j++)
    for (const cubet &d : frames[j])
      if (std::includes(c.begin(), c.end(), d.begin(), d.end()))
        return true;
  return false;
}

void ts_pdrt::add_blocked(const cubet &c, unsigned level)
{
  frames[level].push_back(c);
  solver->assert_expr(
    implies2tc(level_lits[level], not2tc(cube_expr(c, false))));
}

bool ts_pdrt::relative_inductive(
  const cubet &c,
  unsigned level,
  cubet &core)
{
  std::vector<expr2tc> assumptions = frame(level - 1);
  assumptions.push_back(act_trans);
  assumptions.push_back(not2tc(cube_expr(c, false)));
  std::unordered_map<expr2tc, unsigned, irep2_hash> index;
  for (unsigned i = 0; i < c.size(); i++)
  {
    expr2tc lit = literal(c[i], true);
    index.emplace(lit, i);
    assumptions.push_back(lit);
  }
  if (solve(assumptions, "relative-induction") == smt_resultt::P_SATISFIABLE)
    return false;

  core.clear();
  for (const expr2tc &u : solver->unsat_assumptions())
  {
    auto it = index.find(u);
    if (it != index.end())
      core.push_back(c[it->second]);
  }
  std::sort(core.begin(), core.end());
  return true;
}

ts_pdrt::cubet
ts_pdrt::generalize(const cubet &c, unsigned level, const cubet &core)
{
  // A smaller cube still blocks at this level: its complement is contained
  // in !c, so F & !g & T & g' is implied UNSAT by the query that gave the core.
  cubet g = core;
  for (const litt &l : c)
  {
    if (!intersects_init(g))
      break;
    if (!std::binary_search(g.begin(), g.end(), l))
    {
      g.push_back(l);
      std::sort(g.begin(), g.end());
    }
  }

  unsigned attempts = 0;
  for (size_t i = 0; g.size() <= drop_max_cube && i < g.size() &&
                     attempts < drop_budget;)
  {
    cubet candidate = g;
    candidate.erase(candidate.begin() + i);
    if (candidate.empty() || intersects_init(candidate))
    {
      i++;
      continue;
    }
    attempts++;
    cubet smaller;
    if (relative_inductive(candidate, level, smaller))
      g = !smaller.empty() && !intersects_init(smaller) ? smaller : candidate;
    else
      i++;
  }
  return g;
}

bool ts_pdrt::block(obligationt bad, unsigned &cex_depth)
{
  std::multimap<unsigned, obligationt> queue;
  queue.emplace(bad.level, bad);
  while (!queue.empty())
  {
    obligationt ob = queue.begin()->second;
    queue.erase(queue.begin());

    if (ob.level == 0 || intersects_init(ob.cube))
    {
      cex_depth = ob.steps_to_bad;
      return false;
    }
    if (subsumed(ob.cube, ob.level))
      continue;

    cubet core;
    if (relative_inductive(ob.cube, ob.level, core))
    {
      cubet g = generalize(ob.cube, ob.level, core);
      unsigned level = ob.level;
      cubet ignored;
      while (level + 1 < frames.size() &&
             relative_inductive(g, level + 1, ignored))
        level++;
      add_blocked(g, level);
      if (level + 1 < frames.size())
        queue.emplace(level + 1, obligationt{level + 1, ob.cube, ob.steps_to_bad});
    }
    else
    {
      cubet pred =
        lift(model_cube(), and2tc(trans_expr, cube_expr(ob.cube, true)));
      queue.emplace(
        ob.level - 1, obligationt{ob.level - 1, pred, ob.steps_to_bad + 1});
      queue.emplace(ob.level, ob);
    }
  }
  return true;
}

unsigned ts_pdrt::propagate()
{
  for (unsigned i = 1; i + 1 < frames.size(); i++)
  {
    std::vector<cubet> stay;
    std::vector<cubet> current = frames[i];
    for (const cubet &c : current)
    {
      std::vector<expr2tc> assumptions = frame(i);
      assumptions.push_back(act_trans);
      for (const litt &l : c)
        assumptions.push_back(literal(l, true));
      if (solve(assumptions, "propagate") == smt_resultt::P_UNSATISFIABLE)
        add_blocked(c, i + 1);
      else
        stay.push_back(c);
    }
    frames[i] = stay;
    if (stay.empty())
      return i;
  }
  return 0;
}

bool ts_pdrt::check_invariant(unsigned from_level)
{
  std::vector<cubet> cubes;
  for (unsigned j = from_level; j < frames.size(); j++)
    cubes.insert(cubes.end(), frames[j].begin(), frames[j].end());

  // A second solver, so the check does not rely on how frames were encoded.
  ts_pdrt fresh(ts, ns, *options_);
  if (!fresh.build_bits())
    return false;
  std::vector<expr2tc> cur, next;
  for (const cubet &c : cubes)
  {
    cur.push_back(not2tc(fresh.cube_expr(c, false)));
    next.push_back(not2tc(fresh.cube_expr(c, true)));
  }
  expr2tc inv = conjunction(cur), inv_next = conjunction(next);

  const bool initiation =
    fresh.solve({fresh.act_init, fresh.act_inv, not2tc(inv)}, "check") ==
    smt_resultt::P_UNSATISFIABLE;
  const bool consecution =
    fresh.solve(
      {fresh.act_inv, fresh.act_trans, inv, not2tc(inv_next)}, "check") ==
    smt_resultt::P_UNSATISFIABLE;
  const bool safety =
    fresh.solve({fresh.act_inv, inv, fresh.bad_literal(0)}, "check") ==
    smt_resultt::P_UNSATISFIABLE;
  log_status(
    "PDR invariant: {} clauses; initiation {}, consecution {}, safety {}",
    cubes.size(),
    initiation ? "holds" : "FAILS",
    consecution ? "holds" : "FAILS",
    safety ? "holds" : "FAILS");
  return initiation && consecution && safety;
}

ts_pdrt::resultt ts_pdrt::run(uint64_t max_frames, unsigned &cex_depth)
{
  try
  {
    if (!build_bits())
      return resultt::unknown;
    if (!check_prefix())
    {
      cex_depth = 0;
      return resultt::unsafe;
    }
    if (ts.bad.empty())
      return resultt::safe;
    if (
      solve({act_init, act_inv, bad_literal(0)}, "bad") ==
      smt_resultt::P_SATISFIABLE)
    {
      cex_depth = 0;
      return resultt::unsafe;
    }

    frames.resize(2);
    level_lits = {expr2tc(), symbol2tc(get_bool_type(), "ts$pdr$level$1")};
    log_status("PDR over {} state bits", bit_of.size());

    while (frames.size() - 1 <= max_frames)
    {
      fine_timet start = current_time();
      unsigned k = frames.size() - 1;
      while (true)
      {
        std::vector<expr2tc> assumptions = frame(k);
        assumptions.push_back(bad_literal(0));
        if (solve(assumptions, "bad") == smt_resultt::P_UNSATISFIABLE)
          break;
        if (!block({k, lift(model_cube(), bad_expr), 0}, cex_depth))
          return resultt::unsafe;
      }

      frames.emplace_back();
      level_lits.push_back(
        symbol2tc(get_bool_type(), "ts$pdr$level$" + std::to_string(k + 1)));
      unsigned fixpoint = propagate();

      std::string sizes, counts;
      for (unsigned i = 1; i < frames.size(); i++)
        sizes += " " + std::to_string(frames[i].size());
      for (const auto &[kind, stat] : query_stats)
        counts += fmt::format(" {}={}/{:.1f}s", kind, stat.first, stat.second);
      log_status(
        "PDR frame {}: clauses per level{}; {} queries:{} ({}s)",
        k,
        sizes,
        queries,
        counts,
        time2string(current_time() - start));

      if (fixpoint)
      {
        if (check_invariant(fixpoint + 1))
          return resultt::safe;
        reason = "the invariant PDR found failed its independent check";
        return resultt::unknown;
      }
    }
    reason = "frame limit reached";
    return resultt::unknown;
  }
  catch (const std::runtime_error &e)
  {
    reason = e.what();
    return resultt::unknown;
  }
}
