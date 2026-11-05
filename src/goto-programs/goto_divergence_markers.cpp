#include <goto-programs/goto_divergence_markers.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <algorithm>
#include <deque>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

const char *const divergence_marker_comment =
  "non-terminating execution: a reachable loop can run forever";

namespace
{
using function_sett = std::unordered_set<irep_idt, irep_id_hash>;

/// The source name of a function: `exit` for `c:@F@exit`, `terminate` for
/// `c:@N@std@F@terminate#`, `f` for a static `c:file.c@F@f`.
std::string base_name(const irep_idt &id)
{
  std::string name = id.as_string();
  const std::size_t f = name.rfind("@F@");
  if (f != std::string::npos)
    name = name.substr(f + 3);
  return name.substr(0, name.find('#'));
}

/// Calls that end the program or the path. Matched by name: several have no
/// body in GOTO, or an empty one under --no-assertions, although the real
/// program ends there.
bool is_abort_like(const irep_idt &id)
{
  static const std::unordered_set<std::string> names = {
    "abort",
    "exit",
    "_Exit",
    "quick_exit",
    "__assert_fail",
    "__assert_rtn",
    "__assert",
    "err",
    "errx",
    "verr",
    "verrx",
    "__stack_chk_fail",
    "terminate",
    "raise",
    "kill",
    "__VERIFIER_error",
    "reach_error",
    "__builtin_trap",
    "__builtin_unreachable",
    "pthread_exit",
    "longjmp",
    "_longjmp",
    "siglongjmp",
    "__ESBMC_assume",
    "__VERIFIER_assume",
  };
  return names.count(base_name(id)) != 0;
}

/// The symbol a simplified branch guard tests when both outcomes are possible
/// for any value of the symbol's type: `s`, `s == c` or `s != c`.
std::optional<irep_idt> tested_symbol(const expr2tc &guard)
{
  if (is_symbol2t(guard) && (is_bool_type(guard) || is_bv_type(guard)))
    return to_symbol2t(guard).thename;

  expr2tc side1, side2;
  if (is_equality2t(guard))
  {
    side1 = to_equality2t(guard).side_1;
    side2 = to_equality2t(guard).side_2;
  }
  else if (is_notequal2t(guard))
  {
    side1 = to_notequal2t(guard).side_1;
    side2 = to_notequal2t(guard).side_2;
  }
  else
    return std::nullopt;

  if (is_symbol2t(side2))
    std::swap(side1, side2);
  if (
    is_symbol2t(side1) && (is_bv_type(side1) || is_bool_type(side1)) &&
    (is_constant_int2t(side2) || is_constant_bool2t(side2)))
    return to_symbol2t(side1).thename;
  return std::nullopt;
}

bool is_nondet(const expr2tc &e)
{
  return is_sideeffect2t(e) &&
         to_sideeffect2t(e).kind == sideeffect_allockind::nondet;
}

/// True when @p var holds, at @p branch, a value assigned from a nondet on
/// every path into it, with nothing in between that could constrain it. Only
/// __VERIFIER_nondet_*() gives such a value: a library function without a
/// body, such as getpid(), has a result the program cannot choose.
bool is_fresh_nondet(
  const goto_programt &body,
  goto_programt::const_targett branch,
  irep_idt var)
{
  if (branch->is_target())
    return false;

  constexpr int max_walk = 32;
  auto p = branch;
  for (int steps = 0; steps < max_walk && p != body.instructions.begin();
       ++steps)
  {
    --p;
    if (p->is_assign())
    {
      const code_assign2t &assign = to_code_assign2t(p->code);
      if (!is_symbol2t(assign.target))
        return false;
      if (to_symbol2t(assign.target).thename == var)
      {
        // An uninitialised local reads an indeterminate value, not a choice.
        if (is_nondet(assign.source))
          return !p->location.get_bool("#decl_nondet");
        if (
          !is_symbol2t(assign.source) ||
          assign.source->type != assign.target->type)
          return false;
        var = to_symbol2t(assign.source).thename;
      }
    }
    else if (!(p->is_skip() || p->is_location() || p->type == DEAD ||
               (p->is_decl() && to_code_decl2t(p->code).value != var)))
      return false;

    // Another path could enter here, past the nondet.
    if (p->is_target())
      return false;
  }
  return false;
}

/// The outcomes of a branch guard the chooser can force by picking the fresh
/// nondets it tests. In `a && b` forcing `a` false forces the conjunction
/// false; forcing it true would need `a` and `b` independent, which is not
/// checked.
struct forcingt
{
  bool to_true = false;
  bool to_false = false;
};

forcingt forcing(
  const expr2tc &guard,
  const goto_programt &body,
  goto_programt::const_targett branch)
{
  if (is_not2t(guard))
  {
    const forcingt f = forcing(to_not2t(guard).value, body, branch);
    return {f.to_false, f.to_true};
  }
  if (is_and2t(guard))
  {
    const auto &a = to_and2t(guard);
    return {
      false,
      forcing(a.side_1, body, branch).to_false ||
        forcing(a.side_2, body, branch).to_false};
  }
  if (is_or2t(guard))
  {
    const auto &o = to_or2t(guard);
    return {
      forcing(o.side_1, body, branch).to_true ||
        forcing(o.side_2, body, branch).to_true,
      false};
  }
  const auto var = tested_symbol(guard);
  if (var && is_fresh_nondet(body, branch, *var))
    return {true, true};
  return {};
}

/// A safety game on one function's control-flow graph. The chooser picks the
/// outcome of fresh nondet branches; every other branch, and every call's
/// eventual return, is up to the program state. A node is winning when the
/// chooser can keep the execution away from ASSUMEs, abort-like calls, calls
/// to functions not in @p total, and (unless @p returns_ok) the function's
/// exit, forever.
class divergence_gamet
{
public:
  divergence_gamet(
    const goto_functionst &goto_functions,
    const goto_programt &body,
    const function_sett &total,
    bool returns_ok)
  {
    for (auto it = body.instructions.begin(); it != body.instructions.end();
         ++it)
    {
      index.emplace(&*it, nodes.size());
      nodes.push_back(it);
    }

    const size_t n = nodes.size();
    std::vector<std::vector<size_t>> preds(n);
    succs.assign(n, {});
    force.assign(n, {});
    win.assign(n, true);

    std::deque<size_t> lost;
    for (size_t i = 0; i < n; ++i)
    {
      const auto it = nodes[i];
      bool bad = false;
      if (it->is_end_function() || it->is_return())
        bad = !returns_ok;
      else if (it->is_goto())
      {
        expr2tc guard = it->guard;
        simplify(guard);
        if (!is_false(guard))
          for (auto t : it->targets)
            succs[i].push_back(index.at(&*t));
        if (!is_true(guard))
        {
          succs[i].push_back(i + 1);
          if (succs[i].size() == 2 && succs[i][0] != succs[i][1])
            force[i] = forcing(guard, body, it);
        }
      }
      else
      {
        bad = !passable(*it, body.hide, goto_functions, total);
        succs[i].push_back(i + 1);
      }

      for (size_t s : succs[i])
        if (s >= n)
          bad = true;
      if (bad)
      {
        win[i] = false;
        lost.push_back(i);
        continue;
      }
      for (size_t s : succs[i])
        preds[s].push_back(i);
    }

    while (!lost.empty())
    {
      const size_t l = lost.front();
      lost.pop_front();
      for (size_t p : preds[l])
      {
        if (!win[p] || holds(p))
          continue;
        win[p] = false;
        lost.push_back(p);
      }
    }
  }

  bool wins_at(goto_programt::const_targett it) const
  {
    return win.at(index.at(&*it));
  }

private:
  /// For a two-way branch, succs is {taken, fall-through}.
  bool holds(size_t i) const
  {
    const auto &next = succs[i];
    if (next.size() == 2 && next[0] != next[1])
    {
      const bool taken = win[next[0]], fall = win[next[1]];
      return (taken && fall) || (force[i].to_true && taken) ||
             (force[i].to_false && fall);
    }
    return std::all_of(
      next.begin(), next.end(), [this](size_t s) { return win[s]; });
  }

  static bool passable(
    const goto_programt::instructiont &inst,
    bool in_library,
    const goto_functionst &goto_functions,
    const function_sett &total)
  {
    switch (inst.type)
    {
    case ASSUME:
    {
      expr2tc guard = inst.guard;
      simplify(guard);
      return is_true(guard);
    }
    // A user assertion is the program's own abort; one in an operational
    // model checks for undefined behaviour and does not stop the program.
    case ASSERT:
      return in_library;
    case FUNCTION_CALL:
    {
      const code_function_call2t &call = to_code_function_call2t(inst.code);
      if (!is_symbol2t(call.function))
        return false;
      const irep_idt &id = to_symbol2t(call.function).thename;
      if (is_abort_like(id))
        return false;
      auto it = goto_functions.function_map.find(id);
      if (it != goto_functions.function_map.end() && it->second.body_available)
        return total.count(id) != 0;
      return base_name(id).rfind("__ESBMC", 0) != 0;
    }
    case OTHER:
      return is_code_expression2t(inst.code) || is_code_printf2t(inst.code) ||
             is_code_free2t(inst.code);
    case THROW:
    case CATCH:
      return false;
    default:
      return true;
    }
  }

  std::vector<goto_programt::const_targett> nodes;
  std::unordered_map<const goto_programt::instructiont *, size_t> index;
  std::vector<std::vector<size_t>> succs;
  std::vector<forcingt> force;
  std::vector<bool> win;
};

/// The functions from whose entry the chooser can always either return or
/// run forever: the greatest fixpoint, so unbounded recursion counts as
/// running forever.
function_sett total_functions(const goto_functionst &goto_functions)
{
  function_sett total;
  for (const auto &[id, f] : goto_functions.function_map)
    if (f.body_available && !is_abort_like(id))
      total.insert(id);

  bool changed = true;
  while (changed)
  {
    changed = false;
    for (const auto &[id, f] : goto_functions.function_map)
    {
      if (!total.count(id))
        continue;
      const divergence_gamet game(goto_functions, f.body, total, true);
      if (!game.wins_at(f.body.instructions.begin()))
      {
        total.erase(id);
        changed = true;
      }
    }
  }
  return total;
}

unsigned insert_markers(goto_functionst &goto_functions)
{
  const function_sett total = total_functions(goto_functions);
  unsigned markers = 0;
  for (auto &[id, f] : goto_functions.function_map)
  {
    if (!f.body_available || f.body.hide || id == "__ESBMC_main")
      continue;

    goto_programt &body = f.body;
    std::vector<goto_programt::targett> heads;
    {
      const divergence_gamet game(goto_functions, body, total, false);
      for (auto it = body.instructions.begin(); it != body.instructions.end();
           ++it)
        if (it->is_backwards_goto())
          for (auto t : it->targets)
            if (
              game.wins_at(t) &&
              std::find(heads.begin(), heads.end(), t) == heads.end())
              heads.push_back(t);
    }

    for (auto head : heads)
    {
      goto_programt::instructiont marker;
      marker.type = ASSERT;
      marker.guard = gen_false_expr();
      marker.location = head->location;
      marker.location.user_provided(false);
      marker.location.comment(divergence_marker_comment);
      marker.function = head->function;
      body.insert_swap(head, marker);
      ++markers;
    }
  }
  return markers;
}

/// Another thread could end the program while a loop runs forever, so a
/// reachable divergent loop proves nothing there.
bool creates_threads(const goto_functionst &goto_functions)
{
  for (const auto &[id, f] : goto_functions.function_map)
    for (const auto &inst : f.body.instructions)
    {
      if (!inst.is_function_call())
        continue;
      const expr2tc &callee = to_code_function_call2t(inst.code).function;
      if (
        is_symbol2t(callee) &&
        base_name(to_symbol2t(callee).thename) == "pthread_create")
        return true;
    }
  return false;
}
} // namespace

std::unique_ptr<goto_functionst>
make_divergence_program(const goto_functionst &goto_functions)
{
  if (creates_threads(goto_functions))
    return nullptr;

  auto program = std::make_unique<goto_functionst>(goto_functions);
  program->update();
  if (insert_markers(*program) == 0)
    return nullptr;

  // Only the markers are properties here, but a failing user assertion or an
  // abort-like call still ends the path it is on.
  for (auto &[id, f] : program->function_map)
    for (auto &inst : f.body.instructions)
    {
      // An undefined extern variable is havoced; keep only its zero value,
      // which makes markers no easier to reach.
      if (id == "__ESBMC_main" && inst.is_assign())
      {
        code_assign2t &assign = to_code_assign2t(inst.code);
        if (
          is_nondet(assign.source) &&
          (is_bv_type(assign.target) || is_pointer_type(assign.target) ||
           is_bool_type(assign.target)))
          assign.source = gen_zero(assign.target->type);
      }

      if (
        inst.is_assert() &&
        inst.location.comment().as_string() != divergence_marker_comment)
      {
        const expr2tc condition = inst.guard;
        if (f.body.hide)
          inst.make_skip();
        else
          inst.make_assumption(condition);
      }
      else if (inst.is_function_call())
      {
        const expr2tc &callee = to_code_function_call2t(inst.code).function;
        if (is_symbol2t(callee) && is_abort_like(to_symbol2t(callee).thename))
          inst.make_assumption(gen_false_expr());
      }
    }
  program->update();
  return program;
}
