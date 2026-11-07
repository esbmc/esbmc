#include <esbmc/non_termination.h>

#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <solvers/solve.h>
#include <solvers/smt/smt_conv.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <util/lang/c_types.h>
#include <util/irep/std_expr.h>
#include <util/base/yaml_parser.h>
#include <util/arith/mp_arith.h>
#include <util/base/i2string.h>
#include <util/config/config.h>

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
/// Discharge a candidate non-terminating execution as an SMT
/// satisfiability query against a fresh solver. Returns true iff @p
/// formula is SATISFIABLE (i.e. the demon really CAN keep the loop
/// alive on the candidate path). UNSAT or UNKNOWN return false — the
/// detector then refuses to flag non-termination, keeping the policy
/// conservative on the wrong-false side.
bool is_sat(const expr2tc &formula, optionst &options, const namespacet &ns)
{
  std::unique_ptr<smt_convt> solver(create_solver("", ns, options));
  solver->assert_expr(formula);
  return solver->dec_solve() == P_SATISFIABLE;
}

/// True if @p e has any subexpression that we can't safely hand to the
/// solver (dereference, array index, struct/member op, byte op, opaque
/// sideeffect). Used to reject IF guards in the callee body that touch
/// memory — the eca-rers2012 shape we target uses only scalar globals,
/// so any guard that doesn't fit that shape disqualifies the loop.
bool touches_memory_or_sideeffect(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (
    is_dereference2t(e) || is_index2t(e) || is_member2t(e) ||
    is_byte_extract2t(e) || is_byte_update2t(e) || is_sideeffect2t(e))
    return true;
  bool found = false;
  e->foreach_operand([&](const expr2tc &op) {
    if (!found)
      found = touches_memory_or_sideeffect(op);
  });
  return found;
}

/// True iff @p guard is a literal `false` constant — the shape the
/// frontend produces from C's `while(1)` (the IF says `IF false GOTO
/// exit`, which never fires; control falls through to the body).
bool is_constant_false(const expr2tc &guard)
{
  if (is_nil_expr(guard))
    return false;
  if (is_false(guard))
    return true;
  if (is_not2t(guard) && is_true(to_not2t(guard).value))
    return true;
  return false;
}

/// Recognise a set-membership predicate `s ∈ {v1, ..., vn}` on a
/// single symbol @p s and constant integer values. Two semantically
/// equivalent shapes are accepted:
///
///   (a) `!(s != v1 && ... && s != vn)` — the C-source shape:
///       a logical-OR of equalities, expressed via De Morgan on a
///       conjunction of disequalities.
///   (b) `s == v1 || ... || s == vn` — the form interval analysis
///       and other simplifiers tend to produce.
///
/// On success, @p out_symbol holds @p s (each disjunct must reference
/// the same symbol) and @p out_values receives the list of constants
/// in source order. Returns false if the predicate isn't recognisable
/// — wrong top-level shape, mixed symbols across operands, an operand
/// that isn't `s OP const`, or an empty operand list.
///
/// The two shapes denote the same set; structural recognisers that
/// care about "which value the input takes" should call this once
/// rather than hard-coding either form.
bool extract_value_set_membership(
  const expr2tc &guard,
  expr2tc &out_symbol,
  std::vector<BigInt> &out_values)
{
  if (is_nil_expr(guard))
    return false;

  std::vector<expr2tc> atoms;
  bool
    atom_is_neq; // true ⇒ shape (a) (atoms are !=), false ⇒ (b) (atoms are ==)

  if (is_not2t(guard))
  {
    expr2tc inner = to_not2t(guard).value;
    atom_is_neq = true;
    std::function<void(const expr2tc &)> flatten = [&](const expr2tc &e) {
      if (is_and2t(e))
      {
        flatten(to_and2t(e).side_1);
        flatten(to_and2t(e).side_2);
      }
      else
        atoms.push_back(e);
    };
    flatten(inner);
  }
  else
  {
    atom_is_neq = false;
    std::function<void(const expr2tc &)> flatten = [&](const expr2tc &e) {
      if (is_or2t(e))
      {
        flatten(to_or2t(e).side_1);
        flatten(to_or2t(e).side_2);
      }
      else
        atoms.push_back(e);
    };
    flatten(guard);
  }
  if (atoms.empty())
    return false;

  out_values.clear();
  out_symbol = expr2tc();
  for (const expr2tc &a : atoms)
  {
    expr2tc lhs, rhs;
    if (atom_is_neq)
    {
      if (!is_notequal2t(a))
        return false;
      lhs = to_notequal2t(a).side_1;
      rhs = to_notequal2t(a).side_2;
    }
    else
    {
      if (!is_equality2t(a))
        return false;
      lhs = to_equality2t(a).side_1;
      rhs = to_equality2t(a).side_2;
    }
    if (is_constant_int2t(lhs) && is_symbol2t(rhs))
      std::swap(lhs, rhs);
    if (!is_symbol2t(lhs) || !is_constant_int2t(rhs))
      return false;
    if (is_nil_expr(out_symbol))
      out_symbol = lhs;
    else if (
      !is_symbol2t(out_symbol) ||
      to_symbol2t(out_symbol).thename != to_symbol2t(lhs).thename)
      return false;
    out_values.push_back(to_constant_int2t(rhs).value);
  }
  return true;
}

/// Recognise an eca-rers2012 main loop: a `while(1)` whose body is
///
///   DECL input
///   ASSIGN input = NONDET(int)        (possibly twice — frontend quirk)
///   IF !(input != v1 && ... && input != vn) GOTO call_label
///   RETURN <val>
///   call_label: FUNCTION_CALL: output = f(input)
///   [DEAD input]
///   GOTO loop_head
///
/// On success, @p valid_inputs receives {v1, .., vn} and @p callee_name
/// receives the called function's symbol id.
bool recognize_eca_main_loop(
  const loopst &loop,
  expr2tc &input_arg,
  std::vector<BigInt> &valid_inputs,
  irep_idt &callee_name,
  std::vector<expr2tc> &result_writes)
{
  // Use the effective loop head — skips ASSUMEs etc. that
  // --interval-analysis inserts at the back-edge target. See
  // loopst::effective_loop_head documentation.
  goto_programt::const_targett head = loop.effective_loop_head();
  goto_programt::const_targett back = loop.get_original_loop_exit();
  if (head == back || !head->is_goto() || !is_constant_false(head->guard))
    return false;
  if (!back->is_goto() || !back->is_backwards_goto())
    return false;
  expr2tc input_sym;
  irep_idt callee;
  std::vector<BigInt> values;
  for (auto it = std::next(head); it != back; ++it)
  {
    if (
      it->is_skip() || it->type == LOCATION || it->type == DECL ||
      it->type == DEAD)
      continue;
    // Tolerate ASSUMEs (from --interval-analysis). They constrain
    // modified-set values but don't alter the loop's control flow.
    if (it->is_assume())
      continue;
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (
        is_symbol2t(a.target) && is_sideeffect2t(a.source) &&
        to_sideeffect2t(a.source).kind == sideeffect2t::allockind::nondet)
      {
        input_sym = a.target;
        continue;
      }
      // A call result read through a temporary, as `int input = f();` and
      // witness validation produce: `tmp = NONDET; input = tmp`.
      if (
        is_symbol2t(a.target) && is_symbol2t(a.source) &&
        !is_nil_expr(input_sym) &&
        to_symbol2t(a.source).thename == to_symbol2t(input_sym).thename)
      {
        input_sym = a.target;
        continue;
      }
      return false;
    }
    if (
      it->is_goto() && !is_true(it->guard) && !it->is_backwards_goto() &&
      it->targets.size() == 1)
    {
      // Input-validation IF: `IF (input ∈ {v1,...,vn}) GOTO call`.
      // Accept the membership predicate in any of these equivalent
      // shapes the frontend / interval simplifier may produce.
      expr2tc the_input;
      if (!extract_value_set_membership(it->guard, the_input, values))
        return false;
      if (
        is_nil_expr(input_sym) || !is_symbol2t(the_input) ||
        to_symbol2t(input_sym).thename != to_symbol2t(the_input).thename)
        return false;
      auto next = std::next(it);
      if (next == back || !next->is_return())
        return false;
      auto tgt = it->targets.front();
      while (tgt != back && tgt->type == DECL)
        ++tgt;
      if (tgt == back || !tgt->is_function_call())
        return false;
      const code_function_call2t &call = to_code_function_call2t(tgt->code);
      if (!is_symbol2t(call.function))
        return false;
      // The call's single argument must be the input symbol.
      if (call.operands.size() != 1)
        return false;
      expr2tc arg = call.operands[0];
      // Allow trivial typecasts wrapping the argument.
      while (is_typecast2t(arg))
        arg = to_typecast2t(arg).from;
      if (
        !is_symbol2t(arg) ||
        to_symbol2t(arg).thename != to_symbol2t(input_sym).thename)
        return false;
      callee = to_symbol2t(call.function).thename;
      // ret is nil when the call's result is discarded.
      const bool keeps_result = !is_nil_expr(call.ret) && is_symbol2t(call.ret);
      if (keeps_result)
        result_writes.push_back(call.ret);
      it = std::next(tgt);
      // The remainder of the body must be DEAD/LOCATION/SKIP only, plus
      // the copy of a call result kept in a temporary (`output = tmp`).
      for (; it != back; ++it)
      {
        if (
          it->is_skip() || it->type == LOCATION || it->type == DEAD ||
          it->type == DECL)
          continue;
        if (
          !it->is_assign() || !keeps_result ||
          to_code_assign2t(it->code).source != call.ret)
          return false;
        result_writes.push_back(to_code_assign2t(it->code).target);
      }
      break;
    }
    return false;
  }
  if (callee == irep_idt() || is_nil_expr(input_sym) || values.empty())
    return false;
  input_arg = input_sym;
  valid_inputs = std::move(values);
  callee_name = callee;
  return true;
}

/// Harvest constant initial values for globals from `__ESBMC_main`,
/// the synthetic entry point ESBMC inserts before calling C's `main`.
/// `__ESBMC_main` is loop-free and consists of a flat sequence of
/// `ASSIGN global = constant` instructions for every static-lifetime
/// scalar with an initialiser. We take the LATEST written constant
/// value per symbol (so a chain `x = 0; x = 5;` records x → 5, which
/// is the state at the point `main` begins executing).
///
/// Symbols with non-constant initialisers (e.g. derived from other
/// globals, or written by helper-call results) are NOT recorded — the
/// SMT formula will leave them existentially free, which is sound but
/// less precise. This is fine for eca-rers2012, where every state
/// variable has a literal initialiser in the source.
std::unordered_map<irep_idt, BigInt, irep_id_hash>
harvest_initial_state(const goto_functionst &goto_functions)
{
  std::unordered_map<irep_idt, BigInt, irep_id_hash> out;
  auto it = goto_functions.function_map.find("__ESBMC_main");
  if (it == goto_functions.function_map.end())
    return out;
  if (!it->second.body_available)
    return out;
  for (const auto &instr : it->second.body.instructions)
  {
    if (!instr.is_assign())
      continue;
    const code_assign2t &a = to_code_assign2t(instr.code);
    if (!is_symbol2t(a.target) || !is_constant_int2t(a.source))
      continue;
    out[to_symbol2t(a.target).thename] = to_constant_int2t(a.source).value;
  }
  return out;
}

using statet = std::unordered_map<irep_idt, expr2tc, irep_id_hash>;
/// The paths of a symbolic run: each one's guard and state.
using pathst = std::vector<std::pair<expr2tc, statet>>;

/// Thrown when the updater does something the run below does not model.
struct unsupportedt
{
};

/// @p e with every symbol read from @p s.
expr2tc read_in(const expr2tc &e, const statet &s)
{
  if (is_symbol2t(e))
  {
    auto it = s.find(to_symbol2t(e).thename);
    if (it == s.end())
      throw unsupportedt();
    return it->second;
  }
  if (is_address_of2t(e) || touches_memory_or_sideeffect(e))
    throw unsupportedt();
  expr2tc out = e;
  out.get()->Foreach_operand([&](expr2tc &op) {
    if (!is_nil_expr(op))
      op = read_in(op, s);
  });
  return out;
}

/// Merge the paths reaching one instruction. Their guards are disjoint, so an
/// ite per differing symbol merges them; a symbol one path lacks is dropped.
std::pair<expr2tc, statet> join(const pathst &in)
{
  auto [guard, s] = in.front();
  for (auto k = std::next(in.begin()); k != in.end(); ++k)
  {
    for (auto v = s.begin(); v != s.end();)
    {
      auto o = k->second.find(v->first);
      if (o == k->second.end())
      {
        v = s.erase(v);
        continue;
      }
      if (o->second != v->second)
        v->second = if2tc(v->second->type, k->first, o->second, v->second);
      ++v;
    }
    guard = or2tc(guard, k->first);
  }
  return {guard, s};
}

void run_once(
  const goto_functionst &goto_functions,
  const goto_functiont &fn,
  const statet &entry,
  unsigned depth,
  pathst &returns);

/// Run @p call in place, adding to @p out the state after each of the
/// callee's returning paths. exit and abort end the path.
void run_call(
  const goto_functionst &goto_functions,
  const code_function_call2t &call,
  const expr2tc &guard,
  const statet &s,
  unsigned depth,
  pathst &out)
{
  if (!is_symbol2t(call.function))
    throw unsupportedt();
  const irep_idt &name = to_symbol2t(call.function).thename;
  if (name == "c:@F@exit" || name == "c:@F@abort")
    return;
  auto callee = goto_functions.function_map.find(name);
  if (
    callee == goto_functions.function_map.end() ||
    !is_code_type(callee->second.type) ||
    to_code_type(callee->second.type).argument_names.size() !=
      call.operands.size())
    throw unsupportedt();
  statet entry = s;
  for (size_t k = 0; k < call.operands.size(); ++k)
    entry[to_code_type(callee->second.type).argument_names[k]] =
      read_in(call.operands[k], s);
  pathst inner;
  run_once(goto_functions, callee->second, entry, depth + 1, inner);
  for (auto &[g, after] : inner)
  {
    const expr2tc ret = after["#return"];
    after.erase("#return");
    if (!is_nil_expr(call.ret))
    {
      if (!is_symbol2t(call.ret) || is_nil_expr(ret))
        throw unsupportedt();
      after[to_symbol2t(call.ret).thename] = ret;
    }
    out.emplace_back(and2tc(guard, g), std::move(after));
  }
}

/// Run @p fn once from @p entry, adding to @p returns the guard and final
/// state of every path that returns; "#return" holds the returned value. An
/// assume cuts a path. Throws unsupportedt when @p fn loops, touches memory or
/// calls a function without a body.
void run_once(
  const goto_functionst &goto_functions,
  const goto_functiont &fn,
  const statet &entry,
  unsigned depth,
  pathst &returns)
{
  if (!fn.body_available || depth > 8)
    throw unsupportedt();
  std::unordered_map<const goto_programt::instructiont *, pathst> pending;
  pending[&fn.body.instructions.front()].emplace_back(gen_true_expr(), entry);
  for (auto it = fn.body.instructions.begin(); it != fn.body.instructions.end();
       ++it)
  {
    auto p = pending.find(&*it);
    if (p == pending.end() || p->second.empty())
      continue;
    auto [guard, s] = join(p->second);
    pending.erase(p);
    pathst &next = pending[&*std::next(it, !it->is_end_function())];
    switch (it->type)
    {
    case DECL:
      s.erase(to_code_decl2t(it->code).value);
      [[fallthrough]];
    case SKIP:
    case LOCATION:
    case DEAD:
      next.emplace_back(guard, s);
      break;
    case ASSUME:
      next.emplace_back(and2tc(guard, read_in(it->guard, s)), s);
      break;
    case ASSIGN:
      if (!is_symbol2t(to_code_assign2t(it->code).target))
        throw unsupportedt();
      s[to_symbol2t(to_code_assign2t(it->code).target).thename] =
        read_in(to_code_assign2t(it->code).source, s);
      next.emplace_back(guard, s);
      break;
    case GOTO:
    {
      if (it->is_backwards_goto() || it->targets.size() != 1)
        throw unsupportedt();
      const expr2tc c = read_in(it->guard, s);
      pending[&*it->targets.front()].emplace_back(and2tc(guard, c), s);
      if (!is_true(c))
        next.emplace_back(and2tc(guard, not2tc(c)), s);
      break;
    }
    case RETURN:
      if (!is_nil_expr(to_code_return2t(it->code).operand))
        s["#return"] = read_in(to_code_return2t(it->code).operand, s);
      [[fallthrough]];
    case END_FUNCTION:
      returns.emplace_back(guard, s);
      break;
    case FUNCTION_CALL:
      run_call(
        goto_functions,
        to_code_function_call2t(it->code),
        guard,
        s,
        depth,
        next);
      break;
    default:
      throw unsupportedt();
    }
  }
}

/// Control reaches @p head of the entry function @p fname from its start with
/// the globals as __ESBMC_main initialised them: the code before it writes
/// only locals, jumps nowhere past @p head, and neither returns, exits,
/// assumes nor calls anything but a witness assume. Its loops may run
/// forever, but then so does the program.
bool loop_starts_at_initial_state(
  const irep_idt &fname,
  const goto_programt &body,
  goto_programt::const_targett head,
  const namespacet &ns)
{
  if (fname != "c:@F@" + (config.main.empty() ? "main" : config.main))
    return false;
  for (auto it = body.instructions.begin(); it != head; ++it)
  {
    if (
      it->is_skip() || it->type == LOCATION || it->type == DECL ||
      it->type == DEAD)
      continue;
    // A jump past the head could skip the loop.
    if (it->is_goto())
    {
      for (const auto &t : it->targets)
        if (t->location_number > head->location_number)
          return false;
      continue;
    }
    if (
      it->is_function_call() &&
      is_symbol2t(to_code_function_call2t(it->code).function) &&
      to_symbol2t(to_code_function_call2t(it->code).function).thename ==
        "c:@F@__ESBMC_witness_assume")
      continue;
    if (!it->is_assign() || !is_symbol2t(to_code_assign2t(it->code).target))
      return false;
    const symbolt *sym =
      ns.lookup(to_symbol2t(to_code_assign2t(it->code).target).thename);
    if (!sym || sym->static_lifetime)
      return false;
  }
  return true;
}

/// One iteration of the input loop @p loop from @p entry, reading @p in as
/// the input once the body assigns it to @p input_arg: its call to the
/// updater, the assumes around the call, and the copies of the call's result
/// after it. Adds the path after each return to @p out.
void run_iteration(
  const goto_functionst &goto_functions,
  const loopst &loop,
  statet entry,
  const expr2tc &input_arg,
  const expr2tc &in,
  pathst &out)
{
  expr2tc guard = gen_true_expr();
  bool called = false;
  for (auto it = loop.get_original_loop_head();
       it != loop.get_original_loop_exit();
       ++it)
  {
    if (
      it->is_assign() && !called &&
      to_code_assign2t(it->code).target == input_arg)
      entry[to_symbol2t(input_arg).thename] = in;
    else if (it->is_function_call())
    {
      run_call(
        goto_functions,
        to_code_function_call2t(it->code),
        guard,
        entry,
        0,
        out);
      called = true;
    }
    else if (it->is_assume() && !called)
      guard = and2tc(guard, read_in(it->guard, entry));
    else if (it->is_assume())
      for (auto &[g, after] : out)
        g = and2tc(g, read_in(it->guard, after));
    else if (it->is_assign() && called)
    {
      const code_assign2t &copy = to_code_assign2t(it->code);
      if (!is_symbol2t(copy.target))
        throw unsupportedt();
      for (auto &path : out)
        path.second[to_symbol2t(copy.target).thename] =
          read_in(copy.source, path.second);
    }
  }
}

/// Does a run of the input loop from @p s0 repeat a state within @p bound
/// valid inputs, before exit or abort ends it?
///
/// A repeated state proves non-termination: the loop starts in @p s0
/// (loop_starts_at_initial_state), @p s0 is all the state it carries
/// (has_cycle_from_init, and run_once gives up on any read of other state, of
/// memory or of nondet), and run_iteration executes the loop body exactly
/// apart from the input, so each iteration is a function of the state and the
/// input. A state that recurs along returning paths closes a lasso the inputs
/// can follow forever.
bool reaches_cycle_from_init(
  const goto_functionst &goto_functions,
  const loopst &loop,
  const expr2tc &input_arg,
  const std::vector<BigInt> &valid_inputs,
  const statet &s0,
  unsigned bound,
  optionst &options,
  const namespacet &ns)
{
  expr2tc formula = gen_true_expr();
  expr2tc alive = gen_true_expr();
  expr2tc repeats = gen_false_expr();
  std::vector<statet> states{s0};
  for (unsigned t = 0; t < bound; ++t)
  {
    const expr2tc in =
      symbol2tc(input_arg->type, "termination::lasso::input::" + i2string(t));
    expr2tc valid = gen_false_expr();
    for (const BigInt &v : valid_inputs)
      valid = or2tc(valid, equality2tc(in, constant_int2tc(in->type, v)));
    formula = and2tc(formula, valid);

    pathst paths;
    run_iteration(goto_functions, loop, states.back(), input_arg, in, paths);
    expr2tc returned = gen_false_expr();
    for (const auto &path : paths)
      returned = or2tc(returned, path.first);
    alive = and2tc(alive, returned);

    // Name the next state, so that each step reads symbols rather than the
    // whole history. A global is never declared, so every path still has it.
    statet next;
    for (const auto &[name, v0] : s0)
    {
      expr2tc value = states.back().at(name);
      for (const auto &[g, after] : paths)
        value = if2tc(v0->type, g, after.at(name), value);
      next[name] = symbol2tc(
        v0->type,
        "termination::lasso::" + id2string(name) + "::" + i2string(t + 1));
      formula = and2tc(formula, equality2tc(next[name], value));
    }
    for (const statet &earlier : states)
    {
      expr2tc same = alive;
      for (const auto &[name, v] : next)
        same = and2tc(same, equality2tc(earlier.at(name), v));
      repeats = or2tc(repeats, same);
    }
    states.push_back(std::move(next));
  }
  return is_sat(and2tc(formula, repeats), options, ns);
}

/// reaches_cycle_from_init for the loop @p loop of the entry function @p fname,
/// with the globals that have a constant initialiser as its state. The loop
/// itself writes only the input and the call's result (@p result_writes);
/// were one of them such a global, the state would change outside the updater.
bool has_cycle_from_init(
  const goto_functionst &goto_functions,
  const irep_idt &fname,
  const goto_programt &body,
  const loopst &loop,
  const expr2tc &input_arg,
  const std::vector<BigInt> &valid_inputs,
  std::vector<expr2tc> result_writes,
  const std::unordered_map<irep_idt, BigInt, irep_id_hash> &initial_state,
  optionst &options,
  const namespacet &ns)
{
  statet s0;
  for (const auto &[name, value] : initial_state)
  {
    const symbolt *sym = ns.lookup(name);
    if (sym && sym->static_lifetime && is_bv_type(sym->get_type2()))
      s0[name] = constant_int2tc(sym->get_type2(), value);
  }
  result_writes.push_back(input_arg);
  if (
    std::any_of(
      result_writes.begin(),
      result_writes.end(),
      [&](const expr2tc &w) { return s0.count(to_symbol2t(w).thename); }) ||
    !loop_starts_at_initial_state(
      fname, body, loop.get_original_loop_head(), ns))
    return false;
  try
  {
    return reaches_cycle_from_init(
      goto_functions, loop, input_arg, valid_inputs, s0, 24, options, ns);
  }
  catch (const unsupportedt &)
  {
    return false;
  }
  catch (const std::out_of_range &)
  {
    return false;
  }
}

/// The check never reads the witness, so a validator may trust it only for a
/// witness that does nothing but name @p head as the cycle; any other witness
/// is left to symex, which applies it.
bool witness_is_bare_cycle_at(const optionst &options, const locationt &head)
{
  const std::string witness = options.get_option("witness");
  const auto &wps = yaml_parser::get_waypoints(witness);
  return !wps.empty() &&
         std::all_of(wps.begin(), wps.end(), [&](const waypoint &wp) {
           return wp.type == waypoint::assumption &&
                  wp.action == waypoint::cycle &&
                  (wp.value == "1" || wp.value == "true") &&
                  integer2string(wp.line) == head.get_line().as_string();
         });
}

} // namespace

tvt try_prove_non_termination_by_recurrent_set(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns,
  locationt &loop_location)
{
  // Only the eca-rers2012 shape: an input-validation while(1) around one
  // call to an updater. Run the loop from the initial state and look for a
  // state that recurs (has_cycle_from_init); a fixpoint found over all
  // values of the globals could pick a state the program never reaches
  // (#10005).
  const auto initial_state = harvest_initial_state(goto_functions);
  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available || f_it->second.body.hide)
      continue;
    goto_loopst loops(f_it->first, goto_functions, f_it->second);
    for (const auto &loop : loops.get_loops())
    {
      expr2tc input_arg;
      std::vector<BigInt> valid_inputs;
      irep_idt callee_name;
      std::vector<expr2tc> result_writes;
      if (!recognize_eca_main_loop(
            loop, input_arg, valid_inputs, callee_name, result_writes))
        continue;
      const locationt &head = loop.get_original_loop_head()->location;
      if (
        options.get_bool_option("validate-violation-witness") &&
        !witness_is_bare_cycle_at(options, head))
        continue;
      auto callee_it = goto_functions.function_map.find(callee_name);
      if (callee_it == goto_functions.function_map.end())
        continue;
      if (!callee_it->second.body_available || callee_it->second.body.hide)
        continue;
      if (
        has_cycle_from_init(
          goto_functions,
          f_it->first,
          f_it->second.body,
          loop,
          input_arg,
          valid_inputs,
          result_writes,
          initial_state,
          options,
          ns))
      {
        loop_location = head;
        return tvt(tvt::TV_FALSE);
      }
    }
  }
  return tvt(tvt::TV_UNKNOWN);
}
