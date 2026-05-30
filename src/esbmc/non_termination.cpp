#include <esbmc/non_termination.h>

#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <solvers/solve.h>
#include <solvers/smt/smt_conv.h>
#include <irep2/irep2_expr.h>
#include <langapi/language_util.h>
#include <util/c_types.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/std_expr.h>

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace
{
/// True iff @p formula is UNSATISFIABLE under a fresh SMT solver. Mirrors
/// the helper in ranking_synthesis.cpp; replicated here to keep the two
/// passes textually independent. A non-UNSAT result (SAT or unknown) is
/// treated by the recurrent-set check as "this obligation does not
/// discharge", which conservatively means "we cannot prove
/// non-termination via this candidate".
bool is_unsat(const expr2tc &formula, optionst &options, const namespacet &ns)
{
  std::unique_ptr<smt_convt> solver(create_solver("", ns, options));
  solver->assert_expr(formula);
  return solver->dec_solve() == smt_convt::P_UNSATISFIABLE;
}

/// True iff @p e contains a memory-touching subexpression we cannot
/// safely lower to SMT (dereferences, array indices, struct members,
/// byte ops). Used by the recurrent-set check to refuse loops whose
/// exit predicates or update logic involve memory — the MVP only
/// handles purely-scalar finite-state machines.
bool touches_memory(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (
    is_dereference2t(e) || is_index2t(e) || is_member2t(e) ||
    is_byte_extract2t(e) || is_byte_update2t(e))
    return true;
  bool found = false;
  e->foreach_operand([&](const expr2tc &op) {
    if (!found)
      found = touches_memory(op);
  });
  return found;
}

/// A `while(1)`-shaped loop the eca-style recurrent-set check tries to
/// handle. The fields hold the structural pieces extracted by
/// `recognize_eca_loop`.
struct eca_loopt
{
  // The function containing the loop.
  irep_idt function_name;
  // The loop-head IF instruction (guard is constant-false; loop never
  // exits via this IF, only via internal control flow inside the body).
  goto_programt::const_targett head;
  // The back-edge GOTO instruction.
  goto_programt::const_targett back;
  // The (single) NONDET input symbol the body reads at the start of
  // each iteration.
  expr2tc input_symbol;
  // The set of valid input values, harvested from the body's
  // input-validation IF (e.g. `if (input != 2 && ... && input != 6)
  // return -2` yields {2, 3, 4, 5, 6}).
  std::vector<BigInt> valid_inputs;
  // The function called inside the body to perform the state update
  // (e.g. `calculate_output(input)`). Body-available is assumed; if not,
  // recognize_eca_loop refuses the shape.
  irep_idt update_callee;
};

/// True iff @p guard is a literal `false` constant — the shape the
/// frontend produces from C's `while(1)` (the IF says `IF false GOTO
/// exit`, which never fires; control falls through to the body).
bool is_constant_false(const expr2tc &guard)
{
  if (is_nil_expr(guard))
    return false;
  if (is_false(guard))
    return true;
  // Some frontends emit `is_not2t(true)` instead.
  if (is_not2t(guard) && is_true(to_not2t(guard).value))
    return true;
  return false;
}

/// Try to extract the set of valid input values from an IF of the shape
///   IF !(input != v1 && input != v2 && ... && input != vn) THEN GOTO X
/// followed by a RETURN. After negation, `input != v1 && ... && input
/// != vn` is the FALL-THROUGH condition (the path that DOES return).
/// So when the IF jumps (taking the goto), the validation said this
/// input IS one of {v1, ..., vn} — these are the VALID values that
/// continue past the validation. Returns the values in @p out and the
/// input symbol in @p input, or false if the shape doesn't match.
///
/// (We model the original C as `if (input not in S) return; <body>` —
/// the values in S are the only inputs that reach the update logic.)
bool extract_input_validation(
  const expr2tc &if_guard,
  expr2tc &input,
  std::vector<BigInt> &out)
{
  // The IF guard is `!cond` where `cond` is the predicate that
  // identifies REJECTED inputs (`input != 2 && input != 3 && ...`).
  // Peel the outer not.
  if (!is_not2t(if_guard))
    return false;
  expr2tc cond = to_not2t(if_guard).value;
  // cond is a conjunction of `input != const` atoms. Walk it.
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
    // Normalise so the symbol is on the left and the constant on the
    // right.
    if (is_constant_int2t(lhs) && is_symbol2t(rhs))
      std::swap(lhs, rhs);
    if (!is_symbol2t(lhs) || !is_constant_int2t(rhs))
      return false;
    if (is_nil_expr(the_input))
      the_input = lhs;
    else if (
      !is_symbol2t(the_input) ||
      to_symbol2t(the_input).thename != to_symbol2t(lhs).thename)
      return false; // a different symbol appears across conjuncts
    out.push_back(to_constant_int2t(rhs).value);
  }
  input = the_input;
  return true;
}

/// Try to recognise the eca-rers2012 loop shape inside @p fn (named
/// @p fn_name in the function map). Fills @p out and returns true on
/// success. The shape required:
///
///   <loop head:> IF false GOTO past_loop          (the `while(1)`)
///   <maybe DECL input>
///   ASSIGN input = NONDET(int)
///   <maybe ASSIGN input = NONDET(int) again>
///   IF !(input != v1 && ... && input != vn) GOTO 2
///   RETURN -2
///   2: FUNCTION_CALL output = calculate_output(input)
///   <maybe DEAD input>
///   GOTO loop_head                                (back-edge)
///
/// The callee (`calculate_output` above) must be body_available; the
/// recurrent-set check needs to walk its body to find exit predicates
/// and the state update.
bool recognize_eca_loop(
  const irep_idt &fn_name,
  const goto_functiont &fn,
  const loopst &loop,
  const goto_functionst &goto_functions,
  eca_loopt &out)
{
  goto_programt::const_targett head = loop.get_original_loop_head();
  goto_programt::const_targett back = loop.get_original_loop_exit();

  // The head must be an IF whose guard is constant false (the C source
  // `while(1)` lowers to `IF !1 GOTO exit` and `!1` simplifies to false).
  if (!head->is_goto() || !is_constant_false(head->guard))
    return false;
  if (!back->is_goto() || !back->is_backwards_goto())
    return false;

  // Walk the body looking for: an `input = NONDET` (possibly preceded
  // by a DECL), an input-validation IF whose jump target is past a
  // RETURN, a FUNCTION_CALL to a body-available callee, and possibly
  // DEAD instructions and locations.
  expr2tc input_sym;
  std::vector<BigInt> valid_inputs;
  irep_idt callee;
  bool saw_call = false;
  for (auto it = std::next(head); it != back; ++it)
  {
    if (
      it->is_skip() || it->is_location() || it->type == DEAD ||
      it->type == DECL)
      continue;
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (touches_memory(a.target) || touches_memory(a.source))
        return false;
      // We tolerate scalar assignments that read NONDET (the body may
      // emit `input = NONDET; input = NONDET;` due to a frontend
      // peculiarity) or that write a discarded local. We only care
      // about identifying the input symbol, which is the lhs of the
      // last `lhs = NONDET(int)` before the validation IF.
      if (is_symbol2t(a.target) && is_sideeffect2t(a.source))
      {
        if (to_sideeffect2t(a.source).kind == sideeffect2t::allockind::nondet)
          input_sym = a.target;
      }
      continue;
    }
    if (
      it->is_goto() && !is_true(it->guard) && !it->is_backwards_goto() &&
      it->targets.size() == 1)
    {
      // The input-validation IF. Extract valid inputs.
      expr2tc validator_input;
      if (!extract_input_validation(it->guard, validator_input, valid_inputs))
        return false;
      // The validator's input symbol must be the one we just saw.
      if (
        is_nil_expr(input_sym) || !is_symbol2t(validator_input) ||
        to_symbol2t(input_sym).thename != to_symbol2t(validator_input).thename)
        return false;
      // The fall-through must be a RETURN (the "reject" path).
      auto next = std::next(it);
      if (next == back || !next->is_return())
        return false;
      // The jump target should be the FUNCTION_CALL that follows the
      // RETURN.
      auto tgt = it->targets.front();
      if (tgt == back || !tgt->is_function_call())
        return false;
      // The function call must be a direct call to a body-available
      // function.
      const code_function_call2t &call = to_code_function_call2t(tgt->code);
      if (!is_symbol2t(call.function))
        return false;
      auto m =
        goto_functions.function_map.find(to_symbol2t(call.function).thename);
      if (
        m == goto_functions.function_map.end() || !m->second.body_available ||
        m->second.body.hide)
        return false;
      callee = to_symbol2t(call.function).thename;
      saw_call = true;
      // Skip past the call to the back-edge — any remaining
      // instructions are bookkeeping (DEAD input, LOCATION).
      it = std::next(tgt);
      continue;
    }
    if (it->is_function_call())
    {
      // A FUNCTION_CALL we didn't reach via the validation IF: that
      // means there was no validation IF (the body just calls the
      // updater unconditionally). Accept that too, treating S as the
      // full integer domain. NOT implemented in the MVP — bail.
      return false;
    }
    if (it->is_return())
      continue;   // the validation's RETURN, already handled above
    return false; // unrecognised instruction shape
  }
  if (!saw_call || is_nil_expr(input_sym) || valid_inputs.empty())
    return false;

  out.function_name = fn_name;
  out.head = head;
  out.back = back;
  out.input_symbol = input_sym;
  out.valid_inputs = std::move(valid_inputs);
  out.update_callee = callee;
  (void)fn;
  return true;
}

} // namespace

tvt try_prove_non_termination_by_recurrent_set(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns)
{
  // MVP: iterate every function with a body and look for an eca-style
  // recurrent loop. The check is intentionally conservative: a single
  // unrecognised shape returns UNKNOWN. We never return TV_TRUE; either
  // we proved non-termination of some loop (TV_FALSE), or we did not
  // (TV_UNKNOWN).
  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;
    if (f_it->second.body.hide)
      continue;

    goto_loopst loops(f_it->first, goto_functions, f_it->second);
    for (const auto &loop : loops.get_loops())
    {
      eca_loopt el;
      if (!recognize_eca_loop(
            f_it->first, f_it->second, loop, goto_functions, el))
        continue;
      // TODO(next): seed R, refine via whole-R per-input preservation
      // check, verify R disjoint from exit predicates. For now we
      // simply do not certify — the recognition phase lands first and
      // the SMT obligations follow in a subsequent step.
      (void)options;
      (void)ns;
    }
  }
  return tvt(tvt::TV_UNKNOWN);
}
