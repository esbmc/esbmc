#include <transition-system/ts_extraction.h>

#include <goto-symex/equation/symex_target_equation.h>
#include <goto-symex/scheduler/reachability_tree.h>
#include <irep2/irep2_utils.h>
#include <langapi/language_util.h>
#include <util/irep/migrate.h>
#include <util/message/message.h>
#include <util/symtab/symbol.h>

#include <algorithm>
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

bool is_input_symbol(const expr2tc &e)
{
  return is_symbol2t(e) && has_prefix(to_symbol2t(e).thename, "nondet$symex::");
}

/** Memo keys are raw nodes of `e`, so it must not outlive the walk: a freed
 * temporary's address can be recycled by the next allocation.
 */
expr2tc bind_execution_guard_rec(
  const expr2tc &e,
  std::unordered_map<const expr2t *, expr2tc> &memo)
{
  static const irep_idt guard_name = execution_statet::guard_execution_name;
  if (is_nil_expr(e))
    return e;
  if (is_symbol2t(e) && to_symbol2t(e).thename == guard_name)
    return gen_true_expr();
  auto it = memo.find(e.get());
  if (it != memo.end())
    return it->second;
  expr2tc out = e;
  out.get()->Foreach_operand([&memo](expr2tc &op)
                             { op = bind_execution_guard_rec(op, memo); });
  memo.emplace(e.get(), out);
  return out;
}

/** symex guards every claim with the top-level execution guard, a level-1
 * boolean that no SSA step defines. It occurs only positively, and only in
 * p.violated, so binding it to true is exact; left free it becomes another
 * frozen state in the export.
 */
expr2tc bind_execution_guard(const expr2tc &e)
{
  std::unordered_map<const expr2t *, expr2tc> memo;
  expr2tc out = bind_execution_guard_rec(e, memo);
  simplify(out);
  return out;
}

/** Pointers in the state would need their points-to sets preserved across
 * steps; empty aggregates are never assigned, so their markers never fire.
 */
bool unsupported_state_type(const type2tc &t)
{
  if (is_pointer_type(t))
    return true;
  if (is_array_type(t))
    return unsupported_state_type(to_array_type(t).subtype);
  if (!is_struct_type(t) && !is_union_type(t))
    return false;
  const std::vector<type2tc> &members =
    is_struct_type(t) ? to_struct_type(t).members : to_union_type(t).members;
  return members.empty() ||
         std::any_of(members.begin(), members.end(), unsupported_state_type);
}

/** Level-1 identity of an SSA symbol: the variable instance, ignoring its
 * level-2 version.
 */
std::string l1_name(const expr2tc &e)
{
  const symbol2t &s = to_symbol2t(e);
  return s.thename.as_string() + "!" + std::to_string(s.level1_num) + "@" +
         std::to_string(s.thread_num);
}

/** No recursion, and no function pointers outside the library models. Loops in
 * callees are checked during capture: one that would run more than once
 * leaves an unwinding assertion in the step.
 */
bool calls_are_supported(
  const goto_functionst &fns,
  goto_programt::const_targett it,
  goto_programt::const_targett end,
  bool library,
  std::vector<irep_idt> &stack)
{
  for (; it != end; ++it)
  {
    if (!it->is_function_call())
      continue;
    const expr2tc &callee = to_code_function_call2t(it->code).function;
    if (!is_symbol2t(callee))
    {
      if (library)
        continue;
      log_warning("call through a function pointer");
      return false;
    }
    const irep_idt &name = to_symbol2t(callee).thename;
    // src/c2goto/library/setjmp.c models these as __ESBMC_unreachable(): the
    // marker means "not modelled", but with the unreachability intrinsic
    // enabled it becomes a reachable-error property, so every program using
    // them reports a violation that is not in the program. Decline instead.
    if (
      name == "c:@F@setjmp" || name == "c:@F@_setjmp" || name == "c:@F@longjmp")
    {
      log_warning("calls {}, which ESBMC does not model", name.as_string());
      return false;
    }
    if (std::find(stack.begin(), stack.end(), name) != stack.end())
    {
      log_warning("recursion through {}", name.as_string());
      return false;
    }
    auto f = fns.function_map.find(name);
    if (f == fns.function_map.end() || !f->second.body_available)
      continue;
    stack.push_back(name);
    const auto &body = f->second.body;
    if (!calls_are_supported(
          fns,
          body.instructions.begin(),
          body.instructions.end(),
          library || body.hide,
          stack))
      return false;
    stack.pop_back();
  }
  return true;
}

bool recognise(const goto_functionst &fns, loop_shapet &shape)
{
  auto main_fn = fns.function_map.find(main_id);
  if (main_fn == fns.function_map.end() || !main_fn->second.body_available)
  {
    log_warning("no main function");
    return false;
  }
  const goto_programt &body = main_fn->second.body;

  goto_programt::const_targett back = body.instructions.end();
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
    if (it->is_backwards_goto())
    {
      if (back != body.instructions.end())
      {
        log_warning("main has more than one loop");
        return false;
      }
      back = it;
    }
  if (back == body.instructions.end())
  {
    log_warning("main has no loop");
    return false;
  }
  if (!is_true(back->guard))
  {
    log_warning("the loop is not unconditional");
    return false;
  }

  // Leaving the loop (a return, a goto past it, exit or assume(0)) ends the
  // path; capture rejects the program if an assertion can follow.
  goto_programt::const_targett head = back->targets.front();
  for (auto it = head; it != back; ++it)
    if (it->is_throw())
    {
      log_warning("the loop body can throw");
      return false;
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
      // Capture resolves a havocked `*p` to the one object p reaches, or
      // rejects it.
      if (
        !(is_symbol2t(lhs) || is_dereference2t(lhs)) ||
        unsupported_state_type(lhs->type))
      {
        log_warning("havocked state holds a pointer or an empty aggregate");
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
    log_warning("no k-induction havoc before the loop");
    return false;
  }

  std::vector<irep_idt> stack{main_id};
  if (!calls_are_supported(fns, body.instructions.begin(), back, false, stack))
    return false;

  // The entry function runs its own set-up before calling main.
  auto entry_fn = fns.function_map.find(fns.main_id());
  if (entry_fn != fns.function_map.end())
  {
    const auto &insns = entry_fn->second.body.instructions;
    auto call_main = std::find_if(
      insns.begin(),
      insns.end(),
      [](auto &i)
      {
        if (!i.is_function_call())
          return false;
        const expr2tc &f = to_code_function_call2t(i.code).function;
        return is_symbol2t(f) && to_symbol2t(f).thename == main_id;
      });
    std::vector<irep_idt> entry_stack{fns.main_id()};
    if (!calls_are_supported(fns, insns.begin(), call_main, false, entry_stack))
      return false;
  }

  shape.entry = entry->location_number;
  shape.head = head->location_number;
  shape.back = back->location_number;
  return true;
}

goto_programt::targett find_location(goto_programt &body, unsigned number)
{
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
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

/** Insert `marker_i = var_i` before `at`, keeping jumps to `at` on the markers.
 */
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
  e->foreach_operand([&](const expr2tc &op)
                     { collect_symbols(op, out, seen); });
}

int marker_index(const expr2tc &lhs, const std::string &kind)
{
  if (!is_symbol2t(lhs))
    return -1;
  const std::string &name = to_symbol2t(lhs).thename.as_string();
  if (!has_prefix(name, kind))
    return -1;
  return std::stoi(name.substr(kind.size()));
}
} // namespace

/** Extraction marks the three points that delimit a step, runs symex once,
 * then reads the markers back out of the SSA.
 *
 *     x = 0                 prefix
 *     ts$init$0 = x         <- before the havoc: value entering the loop
 *     x = nondet()             the k-induction havoc
 *   head:
 *     ts$pre$0 = x          <- before the head: state at the loop head
 *     assert(x < 4)            body
 *     x = (x + 1) & 3
 *     ts$post$0 = x         <- before the back edge: state at the back edge
 *     goto head
 *
 * One symbolic execution of that yields an SSA in which the marker
 * assignments cut everything into regions:
 *
 *     x@1 == 0              i < init_first        -> prefix_defs
 *     ts$init$0 == x@1      init_first            -> states[0].init
 *     x@2 == nondet         havoc region             dropped (is_only)
 *     ts$pre$0 == x@2       pre_first..pre_last   -> states[0].pre
 *     guard => x@2 < 4      i > pre_last          -> bad
 *     x@3 == (x@2 + 1) & 3                        -> body_defs
 *     ts$post$0 == x@3      post_first            -> states[0].post
 *
 * So each marker's rhs is the value wanted, and the steps between markers are
 * the prefix and the body: assignments become *_defs, assumes become
 * *_assumes (or invariants when inductive-step-only), asserts become *_bad.
 * back_guard is the guard the first ts$post$ step carries.
 */

bool extract_transition_system(
  const goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  bool live_filter,
  transition_systemt &ts)
{
  if (options.get_bool_option("disable-inductive-step"))
  {
    log_warning("the loop's havoc is incomplete (inductive step disabled)");
    return false;
  }
  loop_shapet shape;
  if (!recognise(goto_functions, shape))
    return false;

  goto_functionst program = goto_functions;
  program.update();
  goto_programt &main_body = program.function_map[main_id].body;

  // Introduce the magic markers
  const unsigned n = shape.havoc_vars.size();
  {
    auto back = find_location(main_body, shape.back);
    auto head = find_location(main_body, shape.head);
    auto entry = find_location(main_body, shape.entry);
    insert_markers(main_body, back, context, marker_post, shape.havoc_vars);
    insert_markers(main_body, head, context, marker_pre, shape.havoc_vars);
    insert_markers(main_body, entry, context, marker_init, shape.havoc_vars);
    program.update();
  }
  const auto main_back = std::find_if(
    main_body.instructions.begin(),
    main_body.instructions.end(),
    [](const auto &i) { return i.is_backwards_goto(); });
  // Only the main loop's own unwinding assertion is expected; any other is a
  // callee's, and rejects below.
  const std::string main_unwinding =
    "unwinding assertion loop " + std::to_string(main_back->loop_number);

  optionst opts = options;
  opts.set_option("inductive-step", true);
  opts.set_option("base-case", false);
  opts.set_option("forward-condition", false);
  opts.set_option("partial-loops", false);
  opts.set_option("no-unwinding-assertions", false);
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
  std::shared_ptr<symex_targett> target;
  try
  {
    target = art.get_next_formula().target;
  }
  catch (const inductive_step_disabled_exceptiont &e)
  {
    log_warning("symbolic execution cannot run the step: {}", e.reason);
    return false;
  }
  auto eq = std::dynamic_pointer_cast<symex_target_equationt>(target);

  if (opts.get_bool_option("disable-inductive-step") != was_disabled)
  {
    log_warning("symbolic execution disabled the inductive step");
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
      log_warning("loop markers were not executed exactly once");
      return false;
    }
  if (!(init_first < pre_first && pre_last < post_first))
  {
    log_warning("loop markers out of order");
    return false;
  }
  for (size_t i = 0; i < steps.size(); i++)
  {
    const auto &s = *steps[i];
    if (!s.is_assert())
      continue;
    const std::string &comment = s.comment.as_string();
    if (
      i < post_first &&
      comment.find("unwinding assertion") != std::string::npos)
    {
      log_warning("a loop or recursion inside a step runs more than once");
      return false;
    }
    // Exits end a path, so nothing after the loop may be checked.
    if (i > post_first && comment != main_unwinding)
    {
      log_warning("an assertion is reachable after the loop");
      return false;
    }
  }
  std::unordered_set<expr2tc, irep2_hash> distinct;
  for (unsigned k = 0; k < n; k++)
  {
    if (!is_symbol2t(pre[k]))
    {
      log_warning(
        "state variable {} is a constant at the loop head",
        from_expr(ns, "", shape.havoc_vars[k]));
      return false;
    }
    if (!distinct.insert(pre[k]).second)
    {
      log_warning(
        "state variable {} shares storage with another",
        from_expr(ns, "", shape.havoc_vars[k]));
      return false;
    }
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
      log_warning("the program renumbers dynamic memory");
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
      transition_systemt::propertyt p;
      p.violated = bind_execution_guard(not2tc(implies2tc(a, s.cond)));
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

  transition_systemt::step_localt step_local = body_lhs;
  for (const auto &e : pre)
    step_local.insert(e);
  for (const auto &e : body_syms)
    if (is_input_symbol(e))
      step_local.insert(e);

  std::unordered_set<expr2tc, irep2_hash> prefix_syms;
  std::unordered_set<const expr2t *> prefix_seen;
  for (const auto &e : ts.prefix_defs)
    collect_symbols(e, prefix_syms, prefix_seen);
  for (const auto &e : init)
    collect_symbols(e, prefix_syms, prefix_seen);
  for (const auto &e : step_local)
    if (prefix_syms.count(e))
    {
      log_warning(
        "step variable {} is also defined before the loop",
        from_expr(ns, "", e));
      return false;
    }
  ts.set_step_local(std::move(step_local));

  // A variable that exists before the loop and that the body writes must be
  // state. The havoc misses writes through pointers and ESBMC's allocation
  // bookkeeping; each step would then start from the value before the loop,
  // which symex may even constant-fold, discarding the properties it feeds.
  std::unordered_set<std::string> before_loop, state;
  for (const auto &d : ts.prefix_defs)
    before_loop.insert(l1_name(to_equality2t(d).side_1));
  for (const auto &e : pre)
    state.insert(l1_name(e));
  for (const auto &e : body_lhs)
    if (
      before_loop.count(l1_name(e)) && !state.count(l1_name(e)) &&
      !has_prefix(to_symbol2t(e).thename, "goto_symex::"))
    {
      log_warning(
        "the loop writes {} outside its havocked state", from_expr(ns, "", e));
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
        need(post[st->second]);
        need(init[st->second]);
      }
    }
    auto keep = [&](std::vector<expr2tc> &defs)
    {
      defs.erase(
        std::remove_if(
          defs.begin(),
          defs.end(),
          [&](const expr2tc &e)
          { return !needed.count(to_equality2t(e).side_1); }),
        defs.end());
    };
    keep(ts.prefix_defs);
    keep(ts.body_defs);
  }

  for (unsigned k = 0; k < n; k++)
    if (live[k])
      ts.states.push_back(
        {from_expr(ns, "", shape.havoc_vars[k]), init[k], pre[k], post[k]});

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
