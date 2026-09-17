#include <esbmc/ts/transition_system.h>

#include <goto-symex/equation/symex_target_equation.h>
#include <goto-symex/scheduler/reachability_tree.h>
#include <irep2/irep2_utils.h>
#include <langapi/language_util.h>
#include <util/irep/migrate.h>
#include <util/symtab/symbol.h>

#include <algorithm>
#include <functional>
#include <ostream>
#include <set>

namespace
{
const irep_idt main_id = "c:@F@main";
const std::string marker_init = "ts$init$";
const std::string marker_pre = "ts$pre$";
const std::string marker_post = "ts$post$";

struct loop_shapet
{
  unsigned entry, head, back;
  std::vector<expr2tc> havoc_vars;
};

bool starts_with(const std::string &s, const std::string &prefix)
{
  return s.compare(0, prefix.size(), prefix) == 0;
}

/// `for(;;)` lowers to `IF !1 THEN GOTO exit`.
bool never_taken(const expr2tc &guard)
{
  expr2tc simplified = guard->simplify();
  return is_false(is_nil_expr(simplified) ? guard : simplified);
}

bool is_input_symbol(const expr2tc &e)
{
  return is_symbol2t(e) &&
         starts_with(to_symbol2t(e).thename.as_string(), "nondet$symex::");
}

/// No backward jump in `f` or anything it calls; no function pointers and no
/// recursion.
bool calls_are_loop_free(
  const goto_functionst &fns,
  goto_programt::const_targett it,
  goto_programt::const_targett end,
  std::vector<irep_idt> &stack,
  std::string &reason)
{
  for (; it != end; ++it)
  {
    if (it->is_backwards_goto())
    {
      reason = "a loop is reachable outside the main loop";
      return false;
    }
    if (!it->is_function_call())
      continue;
    const expr2tc &callee = to_code_function_call2t(it->code).function;
    if (!is_symbol2t(callee))
    {
      reason = "call through a function pointer";
      return false;
    }
    const irep_idt &name = to_symbol2t(callee).thename;
    if (std::find(stack.begin(), stack.end(), name) != stack.end())
    {
      reason = "recursion through " + name.as_string();
      return false;
    }
    auto f = fns.function_map.find(name);
    if (f == fns.function_map.end() || !f->second.body_available)
      continue;
    stack.push_back(name);
    const auto &body = f->second.body.instructions;
    if (!calls_are_loop_free(fns, body.begin(), body.end(), stack, reason))
      return false;
    stack.pop_back();
  }
  return true;
}

bool recognise(
  const goto_functionst &fns,
  loop_shapet &shape,
  std::string &reason)
{
  auto main_fn = fns.function_map.find(main_id);
  if (main_fn == fns.function_map.end() || !main_fn->second.body_available)
  {
    reason = "no main function";
    return false;
  }
  const goto_programt &body = main_fn->second.body;

  goto_programt::const_targett back = body.instructions.end();
  for (auto it = body.instructions.begin(); it != body.instructions.end();
       ++it)
    if (it->is_backwards_goto())
    {
      if (back != body.instructions.end())
      {
        reason = "main has more than one loop";
        return false;
      }
      back = it;
    }
  if (back == body.instructions.end())
  {
    reason = "main has no loop";
    return false;
  }
  if (!is_true(back->guard))
  {
    reason = "the loop is not unconditional";
    return false;
  }

  goto_programt::const_targett head = back->targets.front();
  const unsigned lo = head->location_number, hi = back->location_number;
  for (auto it = head; it != back; ++it)
  {
    if (it->is_return() || it->is_throw() || it->is_end_function())
    {
      reason = "the loop body can leave main";
      return false;
    }
    if (it->is_goto() && !never_taken(it->guard))
      for (const auto &t : it->targets)
        if (t->location_number < lo || t->location_number > hi)
        {
          reason = "the loop body has an exit";
          return false;
        }
  }

  goto_programt::const_targett entry = head;
  while (entry != body.instructions.begin())
  {
    auto prev = std::prev(entry);
    if (!prev->inductive_step_instruction)
      break;
    if (prev->is_assign())
    {
      const expr2tc &lhs = to_code_assign2t(prev->code).target;
      if (!is_symbol2t(lhs) || is_pointer_type(lhs->type))
      {
        reason = "havocked state is not a scalar variable";
        return false;
      }
      shape.havoc_vars.insert(shape.havoc_vars.begin(), lhs);
    }
    else if (!prev->is_assume())
      break;
    entry = prev;
  }
  if (shape.havoc_vars.empty())
  {
    reason = "no k-induction havoc before the loop";
    return false;
  }

  std::vector<irep_idt> stack{main_id};
  if (!calls_are_loop_free(
        fns, body.instructions.begin(), back, stack, reason))
    return false;

  // The entry function runs its own set-up before calling main.
  auto entry_fn = fns.function_map.find(fns.main_id());
  if (entry_fn != fns.function_map.end())
  {
    const auto &insns = entry_fn->second.body.instructions;
    auto call_main = std::find_if(insns.begin(), insns.end(), [](auto &i) {
      if (!i.is_function_call())
        return false;
      const expr2tc &f = to_code_function_call2t(i.code).function;
      return is_symbol2t(f) && to_symbol2t(f).thename == main_id;
    });
    std::vector<irep_idt> entry_stack{fns.main_id()};
    if (!calls_are_loop_free(
          fns, insns.begin(), call_main, entry_stack, reason))
      return false;
  }

  shape.entry = entry->location_number;
  shape.head = head->location_number;
  shape.back = back->location_number;
  return true;
}

goto_programt::targett find_location(goto_programt &body, unsigned number)
{
  for (auto it = body.instructions.begin(); it != body.instructions.end();
       ++it)
    if (it->location_number == number)
      return it;
  return body.instructions.end();
}

expr2tc add_marker(
  contextt &context,
  const std::string &kind,
  unsigned index,
  const type2tc &type)
{
  symbolt sym;
  sym.id = kind + std::to_string(index);
  sym.name = sym.id;
  sym.mode = "C";
  sym.set_type(migrate_type_back(type));
  sym.lvalue = true;
  sym.static_lifetime = true;
  sym.is_thread_local = true;
  context.move_symbol_to_context(sym);
  return symbol2tc(type, kind + std::to_string(index));
}

/// Insert `marker_i = var_i` before `at`, keeping jumps to `at` on the markers.
void insert_markers(
  goto_programt &body,
  goto_programt::targett at,
  contextt &context,
  const std::string &kind,
  const std::vector<expr2tc> &vars)
{
  goto_programt markers;
  for (unsigned i = 0; i < vars.size(); i++)
  {
    goto_programt::targett t = markers.add_instruction(ASSIGN);
    t->code =
      code_assign2tc(add_marker(context, kind, i, vars[i]->type), vars[i]);
    t->location = at->location;
    t->function = at->function;
  }
  body.insert_swap(at, markers);
}

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
  e->foreach_operand(
    [&](const expr2tc &op) { collect_symbols(op, out, seen); });
}

int marker_index(const expr2tc &lhs, const std::string &kind)
{
  if (!is_symbol2t(lhs))
    return -1;
  const std::string &name = to_symbol2t(lhs).thename.as_string();
  if (!starts_with(name, kind))
    return -1;
  return std::stoi(name.substr(kind.size()));
}
} // namespace

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

expr2tc transition_systemt::at_step(const expr2tc &e, unsigned step) const
{
  return ts_step_renamert(*this, step)(e);
}

ts_step_renamert::ts_step_renamert(
  const transition_systemt &ts,
  unsigned step)
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
    out.get()->Foreach_operand([&](expr2tc &op) {
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
    [&](const expr2tc &e) -> size_t {
    if (is_nil_expr(e))
      return 0;
    auto it = sizes.find(e.get());
    if (it != sizes.end())
      return it->second;
    size_t n = 1;
    e->foreach_operand(
      [&](const expr2tc &op) { n = std::min(limit + 1, n + tree_size(op)); });
    sizes.emplace(e.get(), n);
    return n;
  };
  auto show = [&](const expr2tc &e) {
    return tree_size(e) > limit ? std::string("<large expression>")
                                : from_expr(ns, "", e);
  };
  out << "Transition system: " << state_names.size() << " state variables, "
      << inputs.size() << " inputs, " << body_defs.size()
      << " step definitions, " << body_assumes.size() << " step assumptions, "
      << bad.size() << " properties, " << invariants.size()
      << " invariants; prefix: " << prefix_defs.size() << " definitions, "
      << prefix_assumes.size() << " assumptions, " << prefix_bad.size()
      << " properties\n";
  out << "State (name: init | pre | post):\n";
  for (unsigned i = 0; i < state_names.size(); i++)
    out << "  " << state_names[i] << ": " << show(state_init[i]) << " | "
        << show(state_pre[i]) << " | " << show(state_post[i]) << "\n";
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

bool extract_transition_system(
  const goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  bool live_filter,
  transition_systemt &ts,
  std::string &reason)
{
  loop_shapet shape;
  if (!recognise(goto_functions, shape, reason))
    return false;

  goto_functionst program = goto_functions;
  program.update();
  goto_programt &main_body = program.function_map[main_id].body;
  auto back = find_location(main_body, shape.back);
  auto head = find_location(main_body, shape.head);
  auto entry = find_location(main_body, shape.entry);
  const unsigned n = shape.havoc_vars.size();
  insert_markers(main_body, back, context, marker_post, shape.havoc_vars);
  insert_markers(main_body, head, context, marker_pre, shape.havoc_vars);
  insert_markers(main_body, entry, context, marker_init, shape.havoc_vars);
  program.update();

  optionst opts = options;
  opts.set_option("inductive-step", true);
  opts.set_option("base-case", false);
  opts.set_option("forward-condition", false);
  opts.set_option("partial-loops", true);
  opts.set_option("no-unwinding-assertions", true);
  opts.set_option("unwind", "1");
  opts.set_option("state-hashing", false);
  opts.set_option("schedule", false);
  opts.set_option("multi-property", false);
  opts.set_option("smt-during-symex", false);
  const bool was_disabled = opts.get_bool_option("disable-inductive-step");

  namespacet ns(context);
  reachability_treet art(
    program, ns, opts, std::make_shared<symex_target_equationt>(ns), context);
  art.setup_for_new_explore();
  auto result = art.get_next_formula();
  auto eq = std::dynamic_pointer_cast<symex_target_equationt>(result.target);

  if (opts.get_bool_option("disable-inductive-step") != was_disabled)
  {
    reason = "symbolic execution disabled the inductive step";
    return false;
  }

  std::vector<const symex_target_equationt::SSA_stept *> steps;
  for (const auto &s : eq->SSA_steps)
    steps.push_back(&s);

  // Marker regions: [init_first, init_last] [pre_first, pre_last]
  // [post_first, ...]
  const size_t none = SIZE_MAX;
  size_t init_first = none, pre_first = none, pre_last = none,
         post_first = none;
  std::vector<expr2tc> init(n), pre(n), post(n);
  std::vector<int> seen_init(n), seen_pre(n), seen_post(n);
  for (size_t i = 0; i < steps.size(); i++)
  {
    const auto &s = *steps[i];
    if (!s.is_assignment())
      continue;
    int k;
    if ((k = marker_index(s.original_lhs, marker_init)) >= 0)
    {
      if (init_first == none)
        init_first = i;
      init[k] = s.rhs;
      seen_init[k]++;
    }
    else if ((k = marker_index(s.original_lhs, marker_pre)) >= 0)
    {
      if (pre_first == none)
        pre_first = i;
      pre_last = i;
      pre[k] = s.rhs;
      seen_pre[k]++;
    }
    else if ((k = marker_index(s.original_lhs, marker_post)) >= 0)
    {
      if (post_first == none)
      {
        post_first = i;
        ts.back_guard = s.guard;
      }
      post[k] = s.rhs;
      seen_post[k]++;
    }
  }
  for (unsigned k = 0; k < n; k++)
    if (seen_init[k] != 1 || seen_pre[k] != 1 || seen_post[k] != 1)
    {
      reason = "loop markers were not executed exactly once";
      return false;
    }
  if (!(init_first < pre_first && pre_last < post_first))
  {
    reason = "loop markers out of order";
    return false;
  }
  for (unsigned k = 0; k < n; k++)
    if (!is_symbol2t(pre[k]))
    {
      reason = "state variable " + from_expr(ns, "", shape.havoc_vars[k]) +
               " is a constant at the loop head";
      return false;
    }

  // Prefix [0, init_first), havoc region (init, pre_first), body
  // (pre_last, post_first).
  std::unordered_set<expr2tc, irep2_hash> body_lhs;
  expr2tc prefix_assumpt = gen_true_expr(), body_assumpt = gen_true_expr();
  for (size_t i = 0; i < post_first; i++)
  {
    const auto &s = *steps[i];
    const bool prefix = i < init_first;
    const bool havoc = i > init_first && i < pre_first;
    const bool body = i > pre_last;
    if (!prefix && !havoc && !body)
      continue;
    if (s.is_renumber())
    {
      reason = "the program renumbers dynamic memory";
      return false;
    }
    if (s.ignore)
      continue;
    const bool is_only = s.source.pc->inductive_step_instruction;
    if (s.is_assignment())
    {
      if (marker_index(s.original_lhs, marker_init) >= 0)
        continue;
      if (havoc && is_only)
        continue;
      if (prefix)
        ts.prefix_defs.push_back(s.cond);
      else
      {
        ts.body_defs.push_back(s.cond);
        body_lhs.insert(s.lhs);
      }
    }
    else if (s.is_assume())
    {
      if (prefix)
      {
        if (is_only)
          continue;
        ts.prefix_assumes.push_back(s.cond);
        prefix_assumpt = and2tc(prefix_assumpt, s.cond);
      }
      else if (is_only)
        ts.invariants.push_back(s.cond);
      else
      {
        ts.body_assumes.push_back(s.cond);
        body_assumpt = and2tc(body_assumpt, s.cond);
      }
    }
    else if (s.is_assert())
    {
      const expr2tc &a = prefix ? prefix_assumpt : body_assumpt;
      ts_propertyt p;
      p.violated = not2tc(implies2tc(a, s.cond));
      p.comment = s.comment.as_string();
      p.location = s.source.pc->location;
      (prefix ? ts.prefix_bad : ts.bad).push_back(p);
    }
  }

  // Step-local: what the body defines, the state at the head, and fresh
  // inputs. Anything else read by the body is fixed for the whole run.
  std::unordered_set<expr2tc, irep2_hash> body_syms;
  std::unordered_set<const expr2t *> seen;
  auto collect = [&](const expr2tc &e) { collect_symbols(e, body_syms, seen); };
  for (const auto &e : ts.body_defs)
    collect(e);
  for (const auto &e : ts.body_assumes)
    collect(e);
  for (const auto &e : ts.invariants)
    collect(e);
  for (const auto &p : ts.bad)
    collect(p.violated);
  collect(ts.back_guard);
  for (const auto &e : post)
    collect(e);

  ts.step_local = body_lhs;
  for (const auto &e : pre)
    ts.step_local.insert(e);
  for (const auto &e : body_syms)
    if (is_input_symbol(e))
      ts.step_local.insert(e);

  std::unordered_set<expr2tc, irep2_hash> prefix_syms;
  std::unordered_set<const expr2t *> prefix_seen;
  for (const auto &e : ts.prefix_defs)
    collect_symbols(e, prefix_syms, prefix_seen);
  for (const auto &e : init)
    collect_symbols(e, prefix_syms, prefix_seen);
  for (const auto &e : ts.step_local)
    if (prefix_syms.count(e))
    {
      reason = "step variable " + from_expr(ns, "", e) +
               " is also defined before the loop";
      return false;
    }

  // Cone of influence: keep what the properties, assumptions, back-edge guard
  // and invariants depend on, following a state variable's head value to its
  // value at the back edge and on entry. Unsliced, ESBMC's memory-model
  // bookkeeping stays in the formula.
  std::vector<bool> live(n, !live_filter);
  if (live_filter)
  {
    std::unordered_map<expr2tc, expr2tc, irep2_hash> def_of;
    for (const auto *defs : {&ts.prefix_defs, &ts.body_defs})
      for (const auto &e : *defs)
        def_of.emplace(to_equality2t(e).side_1, to_equality2t(e).side_2);
    std::unordered_map<expr2tc, unsigned, irep2_hash> state_of;
    for (unsigned k = 0; k < n; k++)
      state_of.emplace(pre[k], k);

    std::unordered_set<expr2tc, irep2_hash> needed;
    std::unordered_set<const expr2t *> visited;
    std::vector<expr2tc> work;
    auto need = [&](const expr2tc &e) {
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
        need(post[st->second]);
        need(init[st->second]);
      }
    }
    auto keep = [&](std::vector<expr2tc> &defs) {
      defs.erase(
        std::remove_if(
          defs.begin(),
          defs.end(),
          [&](const expr2tc &e) {
            return !needed.count(to_equality2t(e).side_1);
          }),
        defs.end());
    };
    keep(ts.prefix_defs);
    keep(ts.body_defs);
  }

  for (unsigned k = 0; k < n; k++)
  {
    if (!live[k])
      continue;
    ts.state_names.push_back(from_expr(ns, "", shape.havoc_vars[k]));
    ts.state_init.push_back(init[k]);
    ts.state_pre.push_back(pre[k]);
    ts.state_post.push_back(post[k]);
  }

  std::set<std::string> input_names;
  for (const auto &s : eq->SSA_steps)
    if (s.is_assignment() && is_input_symbol(s.rhs) && body_syms.count(s.rhs))
      if (input_names.insert(to_symbol2t(s.rhs).thename.as_string()).second)
        ts.inputs.emplace_back(
          s.rhs,
          from_expr(ns, "", s.original_lhs) + " line " +
            s.source.pc->location.get_line().as_string());
  return true;
}
