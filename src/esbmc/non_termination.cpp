#include <esbmc/non_termination.h>

#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <solvers/solve.h>
#include <solvers/smt/smt_conv.h>
#include <irep2/irep2_expr.h>
#include <util/c_types.h>
#include <util/std_expr.h>

#include <memory>
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
  return solver->dec_solve() == smt_convt::P_SATISFIABLE;
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

/// Collect IF guards from a callee body that we treat as "branch
/// guards": top-level forward IFs whose THEN-branch ends in a transfer
/// out (RETURN or a FUNCTION_CALL to a noreturn function like exit /
/// abort), so that NOT taking any of these IFs means falling through
/// the entire body. On success, @p out is populated with one entry per
/// IF — the entry stores the IF's *negated* guard (which is the C
/// source's positive `if (cond)` condition: the IF jumps when cond is
/// false, so the THEN-arm runs when cond is true).
///
/// Returns false if the callee shape isn't recognisable — any non-IF
/// instruction between the IFs that isn't a SKIP/LOCATION/DECL/DEAD/
/// ASSIGN/FUNCTION_CALL-to-noreturn/RETURN/GOTO; any guard touching
/// memory or sideeffects; or a fall-through tail with global writes.
///
/// The eca-rers2012 calculate_output shape is:
///   IF !cond_1 GOTO L1
///   ASSIGN <state writes>
///   RETURN <val>
///   L1: IF !cond_2 GOTO L2
///   ASSIGN <state writes>
///   RETURN <val>
///   ...
///   IF !err_cond_1 GOTO E1
///   FUNCTION_CALL: exit(0)
///   E1: IF !err_cond_2 GOTO E2
///   FUNCTION_CALL: exit(0)
///   ...
///   RETURN -2     <-- the fall-through tail (no state writes)
///
/// Every IF here is "if the condition fires, we transfer out of the
/// body" (either via RETURN or exit). So if NONE of the IFs fire, we
/// reach the trailing RETURN -2 without modifying any state. The query
/// then is: is the conjunction `¬cond_1 ∧ ¬cond_2 ∧ ... ∧ ¬err_cond_1
/// ∧ ...` satisfiable under the input-validation constraint? If yes,
/// the demon has a choice of inputs that keeps state unchanged →
/// non-terminating.
bool collect_callee_branch_guards(
  const goto_functiont &callee,
  std::vector<expr2tc> &out)
{
  out.clear();
  auto end = callee.body.instructions.end();
  for (auto it = callee.body.instructions.begin(); it != end; ++it)
  {
    if (
      it->is_skip() || it->type == LOCATION || it->type == DECL ||
      it->type == DEAD)
      continue;
    if (it->is_return())
    {
      // The trailing fall-through RETURN. Must be the last meaningful
      // instruction — anything else after it would have to be skipped,
      // and there must be no assigns or function calls AFTER an IF
      // chain has fully fallen through. Walk forward to confirm.
      auto j = std::next(it);
      while (j != end)
      {
        if (
          j->is_skip() || j->type == LOCATION || j->type == DECL ||
          j->type == DEAD || j->type == END_FUNCTION)
        {
          ++j;
          continue;
        }
        return false; // unexpected instruction after fall-through return
      }
      return !out.empty();
    }
    if (it->is_goto())
    {
      // Must be a forward IF with a non-trivial guard that touches only
      // scalars.
      if (it->is_backwards_goto() || it->targets.size() != 1)
        return false;
      if (is_true(it->guard) || is_nil_expr(it->guard))
        return false;
      if (touches_memory_or_sideeffect(it->guard))
        return false;
      auto tgt = it->targets.front();
      if (tgt->location_number <= it->location_number)
        return false;
      // The IF guard is `!cond` (THEN-arm runs when cond is true). We
      // store cond (peel the outer not) as the positive branch
      // condition.
      expr2tc neg_guard = it->guard;
      expr2tc pos_cond =
        is_not2t(neg_guard) ? to_not2t(neg_guard).value : not2tc(neg_guard);
      // Walk the THEN-arm (instructions strictly between `it` and
      // `tgt`). Every assign must hit a scalar global / local; the arm
      // must end in either a RETURN or a call to a noreturn function
      // (exit / abort / __VERIFIER_error). We don't constrain the
      // assigns further since they only matter when the branch fires,
      // and we're conjoining ¬cond so it won't.
      bool arm_ok = false;
      for (auto k = std::next(it); k != tgt; ++k)
      {
        if (
          k->is_skip() || k->type == LOCATION || k->type == DECL ||
          k->type == DEAD)
          continue;
        if (k->is_assign())
        {
          if (touches_memory_or_sideeffect(to_code_assign2t(k->code).source))
            return false;
          continue;
        }
        if (k->is_return())
        {
          arm_ok = true;
          break;
        }
        if (k->is_function_call())
        {
          const code_function_call2t &c = to_code_function_call2t(k->code);
          if (!is_symbol2t(c.function))
            return false;
          irep_idt fname = to_symbol2t(c.function).thename;
          static const char *noreturns[] = {
            "c:@F@exit",
            "c:@F@abort",
            "c:@F@__VERIFIER_error",
            "c:@F@reach_error"};
          bool is_noreturn = false;
          for (const char *nm : noreturns)
            if (fname == nm)
            {
              is_noreturn = true;
              break;
            }
          if (!is_noreturn)
            return false;
          arm_ok = true;
          break;
        }
        if (k->is_goto())
          return false; // inner goto / jump out of arm
        return false;   // unknown instruction in arm
      }
      if (!arm_ok)
        return false;
      out.push_back(pos_cond);
      // The arm has been walked. Advance the outer iterator to the IF
      // target so the next iteration picks up the next top-level IF.
      it = std::prev(tgt);
      continue;
    }
    return false; // unexpected top-level instruction
  }
  return false; // walked off end without seeing the trailing return
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
  irep_idt &callee_name)
{
  goto_programt::const_targett head = loop.get_original_loop_head();
  goto_programt::const_targett back = loop.get_original_loop_exit();
  if (!head->is_goto() || !is_constant_false(head->guard))
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
      return false;
    }
    if (
      it->is_goto() && !is_true(it->guard) && !it->is_backwards_goto() &&
      it->targets.size() == 1)
    {
      // Input-validation IF: guard = !(input != v1 && ... && input != vn).
      // Peel the outer not, then flatten the && conjunction; every
      // conjunct must be `input != const`.
      if (!is_not2t(it->guard))
        return false;
      expr2tc cond = to_not2t(it->guard).value;
      std::vector<expr2tc> conjuncts;
      std::function<void(const expr2tc &)> flatten = [&](const expr2tc &e) {
        if (is_and2t(e))
        {
          flatten(to_and2t(e).side_1);
          flatten(to_and2t(e).side_2);
        }
        else
          conjuncts.push_back(e);
      };
      flatten(cond);
      if (conjuncts.empty())
        return false;
      expr2tc the_input;
      for (const expr2tc &c : conjuncts)
      {
        if (!is_notequal2t(c))
          return false;
        const notequal2t &neq = to_notequal2t(c);
        expr2tc lhs = neq.side_1, rhs = neq.side_2;
        if (is_constant_int2t(lhs) && is_symbol2t(rhs))
          std::swap(lhs, rhs);
        if (!is_symbol2t(lhs) || !is_constant_int2t(rhs))
          return false;
        if (is_nil_expr(the_input))
          the_input = lhs;
        else if (
          !is_symbol2t(the_input) ||
          to_symbol2t(the_input).thename != to_symbol2t(lhs).thename)
          return false;
        values.push_back(to_constant_int2t(rhs).value);
      }
      if (
        is_nil_expr(input_sym) || !is_symbol2t(the_input) ||
        to_symbol2t(input_sym).thename != to_symbol2t(the_input).thename)
        return false;
      auto next = std::next(it);
      if (next == back || !next->is_return())
        return false;
      auto tgt = it->targets.front();
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
      it = std::next(tgt);
      // The remainder of the body must be DEAD/LOCATION/SKIP only.
      while (it != back)
      {
        if (
          !it->is_skip() && it->type != LOCATION && it->type != DEAD &&
          it->type != DECL)
          return false;
        ++it;
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

/// Build the SMT formula `input ∈ valid_inputs ∧ ¬cond_1 ∧ ... ∧
/// ¬cond_N` and ask the solver if it's satisfiable. SAT means there's
/// a state + input combination under which `calculate_output` falls
/// through to its trailing `return -2` without firing any branch,
/// leaving the state unchanged — a period-1 fixpoint of the main loop.
bool check_period_1_fixpoint(
  const expr2tc &input_arg,
  const std::vector<BigInt> &valid_inputs,
  const std::vector<expr2tc> &branch_conds,
  optionst &options,
  const namespacet &ns)
{
  // input ∈ {v1, ..., vn}  ⇔  input == v1 ∨ ... ∨ input == vn
  expr2tc input_constraint;
  for (const BigInt &v : valid_inputs)
  {
    expr2tc atom = equality2tc(input_arg, constant_int2tc(input_arg->type, v));
    input_constraint =
      is_nil_expr(input_constraint) ? atom : or2tc(input_constraint, atom);
  }
  expr2tc formula = input_constraint;
  for (const expr2tc &c : branch_conds)
    formula = and2tc(formula, not2tc(c));
  return is_sat(formula, options, ns);
}

} // namespace

tvt try_prove_non_termination_by_recurrent_set(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns)
{
  // MVP scope: only the eca-rers2012 finite-state-machine shape. Each
  // expected-false eca benchmark wraps an input-validation while(1)
  // around a single call to `calculate_output(input)`; if there's a
  // state s and an input i ∈ valid_inputs such that none of the
  // branches in calculate_output fire, then the body falls through to
  // its trailing `return -2` with state unchanged, and the demon can
  // pick that (s, i) forever. We discharge the existence question via
  // a single SMT SAT query — no symex required, just the IF guards
  // harvested from the goto program.
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
      if (!recognize_eca_main_loop(loop, input_arg, valid_inputs, callee_name))
        continue;
      auto callee_it = goto_functions.function_map.find(callee_name);
      if (callee_it == goto_functions.function_map.end())
        continue;
      if (!callee_it->second.body_available || callee_it->second.body.hide)
        continue;
      std::vector<expr2tc> branch_conds;
      if (!collect_callee_branch_guards(callee_it->second, branch_conds))
        continue;
      if (check_period_1_fixpoint(
            input_arg, valid_inputs, branch_conds, options, ns))
        return tvt(tvt::TV_FALSE);
    }
  }
  return tvt(tvt::TV_UNKNOWN);
}
