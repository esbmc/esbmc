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

#include <algorithm>
#include <map>
#include <memory>
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
    // Tolerate ASSUMEs from --interval-analysis. They constrain
    // modified-set values but don't affect the branch structure.
    if (it->is_assume())
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
          j->type == DEAD || j->type == END_FUNCTION || j->is_assume())
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
      if (tgt == end)
        return false; // pathological: jump target is past end-of-function
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
        if (k->is_assume())
          continue; // ASSUME from --interval-analysis; doesn't fire the branch
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
      if (is_symbol2t(call.ret))
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
          !it->is_assign() || !is_symbol2t(call.ret) ||
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

/// Collect the thename of every scalar storage symbol written by any
/// ASSIGN in @p fn into @p out. Used by the soundness pass to identify
/// which globals the callee can mutate across iterations — those are
/// left existentially free in the SMT formula (their reachable set
/// over-approximates), while globals NEVER written are pinned to
/// their initial values (sound: their reachable set IS the singleton
/// initial value).
void collect_writes(
  const goto_functiont &fn,
  std::unordered_set<irep_idt, irep_id_hash> &out)
{
  for (const auto &instr : fn.body.instructions)
  {
    if (!instr.is_assign())
      continue;
    const code_assign2t &a = to_code_assign2t(instr.code);
    if (is_symbol2t(a.target))
      out.insert(to_symbol2t(a.target).thename);
  }
}

/// Replace every leaf `symbol2t` named @p from_name in @p e by a copy
/// of @p to. Used to substitute a callee's formal parameter symbol
/// with the caller's actual-argument symbol so the SMT formula treats
/// them as the same SMT variable.
expr2tc substitute_symbol(
  const expr2tc &e,
  const irep_idt &from_name,
  const expr2tc &to)
{
  if (is_nil_expr(e))
    return e;
  if (is_symbol2t(e) && to_symbol2t(e).thename == from_name)
    return to;
  expr2tc out = e;
  bool changed = false;
  std::vector<expr2tc> new_ops;
  e->foreach_operand([&](const expr2tc &op) {
    expr2tc no = substitute_symbol(op, from_name, to);
    if (no.get() != op.get())
      changed = true;
    new_ops.push_back(no);
  });
  if (!changed)
    return e;
  size_t i = 0;
  out.get()->Foreach_operand([&](expr2tc &op) { op = new_ops[i++]; });
  return out;
}

/// Walk @p e collecting the thename of every `symbol2t` leaf into
/// @p out (deduplicated by name). Used to gather the free state
/// symbols the SMT formula will mention, so we can constrain those
/// that have known initialisers to their initial values.
void collect_free_symbols(
  const expr2tc &e,
  std::unordered_map<irep_idt, expr2tc, irep_id_hash> &out)
{
  if (is_nil_expr(e))
    return;
  if (is_symbol2t(e))
  {
    out.emplace(to_symbol2t(e).thename, e);
    return;
  }
  e->foreach_operand([&](const expr2tc &op) { collect_free_symbols(op, out); });
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

/// Build the SMT formula and ask whether a period-1 fixpoint exists.
///
/// Concretely:
///   input == v1 ∨ ... ∨ input == vn         (caller's input constraint)
///   ∧  state_i == init_i for every state symbol with a known initialiser
///                       AND not modified by any branch of the callee
///   ∧  ¬cond_1 ∧ ... ∧ ¬cond_N               (no branch fires)
///
/// Each `cond_i` has its callee formal parameter substituted by the
/// caller's actual `input_arg` symbol, so all references to "the
/// input" denote the same SMT variable.
///
/// Globals the callee MIGHT modify are left existentially free in the
/// SMT formula — sound because the reachable set of values is at
/// least as wide as their initial value (the demon may have stepped
/// the loop through writing branches in earlier iterations), so any
/// satisfying assignment over reachable states is still a valid
/// non-termination witness from the LASSO closure of that reachable
/// set. Globals NEVER written by the callee, on the other hand, are
/// pinned to their initial value (the reachable set IS the singleton
/// initial value), preventing the wrong-false where the SMT would
/// otherwise pick an unreachable value to falsify a state-dependent
/// guard (e.g. the Codex `a == 1` exploit).
///
/// SAT means there's a state + input combination, consistent with the
/// reachable-state over-approximation, under which the updater falls
/// through to its trailing `return -2` without firing any branch — a
/// real non-terminating execution of the main loop.
bool check_period_1_fixpoint(
  const expr2tc &input_arg,
  const irep_idt &callee_formal_name,
  const std::vector<BigInt> &valid_inputs,
  const std::vector<expr2tc> &branch_conds,
  const std::unordered_map<irep_idt, BigInt, irep_id_hash> &initial_state,
  const std::unordered_set<irep_idt, irep_id_hash> &callee_writes,
  optionst &options,
  const namespacet &ns)
{
  expr2tc input_constraint;
  for (const BigInt &v : valid_inputs)
  {
    expr2tc atom = equality2tc(input_arg, constant_int2tc(input_arg->type, v));
    input_constraint =
      is_nil_expr(input_constraint) ? atom : or2tc(input_constraint, atom);
  }
  // Substitute the callee's formal parameter with the caller's actual
  // input symbol in every branch guard so the SMT formula treats them
  // as the same variable rather than two unrelated free variables.
  std::vector<expr2tc> rewritten;
  rewritten.reserve(branch_conds.size());
  for (const expr2tc &c : branch_conds)
    rewritten.push_back(
      callee_formal_name.empty()
        ? c
        : substitute_symbol(c, callee_formal_name, input_arg));
  // Collect all free symbols appearing in the (rewritten) branch
  // conditions and conjoin known initialisers ONLY for symbols the
  // callee never writes. The caller's input symbol is *deliberately*
  // excluded — its values are constrained by `input_constraint`.
  std::unordered_map<irep_idt, expr2tc, irep_id_hash> free_syms;
  for (const expr2tc &c : rewritten)
    collect_free_symbols(c, free_syms);
  expr2tc formula = input_constraint;
  for (const auto &kv : free_syms)
  {
    if (is_symbol2t(input_arg) && kv.first == to_symbol2t(input_arg).thename)
      continue;
    if (callee_writes.count(kv.first))
      continue; // mutable across iterations — leave free
    auto init = initial_state.find(kv.first);
    if (init == initial_state.end())
      continue;
    expr2tc eq =
      equality2tc(kv.second, constant_int2tc(kv.second->type, init->second));
    formula = and2tc(formula, eq);
  }
  for (const expr2tc &c : rewritten)
    formula = and2tc(formula, not2tc(c));
  return is_sat(formula, options, ns);
}

using statet = std::unordered_map<irep_idt, expr2tc, irep_id_hash>;

/// @p e with every symbol read from @p s; nil when a symbol has no value.
expr2tc read_in(const expr2tc &e, const statet &s)
{
  if (is_symbol2t(e))
  {
    auto it = s.find(to_symbol2t(e).thename);
    return it == s.end() ? expr2tc() : it->second;
  }
  expr2tc out = e;
  bool failed = false;
  out.get()->Foreach_operand([&](expr2tc &op) {
    if (!failed && !is_nil_expr(op))
    {
      op = read_in(op, s);
      failed = is_nil_expr(op);
    }
  });
  return failed ? expr2tc() : out;
}

/// The paths of one call that returned: each one's guard and final state,
/// where "#return" holds the returned value.
struct call_effectt
{
  std::vector<std::pair<expr2tc, statet>> returns;
};

/// Run @p fn once from @p entry symbolically. Its body must be acyclic and
/// write only scalars; a call to a function with a body runs it in place,
/// a call to exit or abort ends the path. False when @p fn has another shape.
bool run_once(
  const goto_functionst &goto_functions,
  const goto_functiont &fn,
  const statet &entry,
  unsigned depth,
  call_effectt &effect)
{
  if (!fn.body_available || depth > 8)
    return false;
  const auto &insns = fn.body.instructions;
  std::unordered_map<const goto_programt::instructiont *, size_t> index;
  for (auto it = insns.begin(); it != insns.end(); ++it)
    index[&*it] = index.size();
  std::vector<std::vector<std::pair<expr2tc, statet>>> pending(insns.size());
  pending[0].emplace_back(gen_true_expr(), entry);

  auto join = [](std::vector<std::pair<expr2tc, statet>> &in) {
    auto [guard, s] = in.front();
    for (size_t k = 1; k < in.size(); ++k)
    {
      const auto &[g, other] = in[k];
      for (auto it = s.begin(); it != s.end();)
      {
        auto o = other.find(it->first);
        if (o == other.end())
          it = s.erase(it);
        else
        {
          if (o->second != it->second)
            it->second = if2tc(it->second->type, g, o->second, it->second);
          ++it;
        }
      }
      guard = or2tc(guard, g);
    }
    return std::make_pair(guard, s);
  };

  size_t i = 0;
  for (auto it = insns.begin(); it != insns.end(); ++it, ++i)
  {
    if (pending[i].empty())
      continue;
    auto [guard, s] = join(pending[i]);
    pending[i].clear();
    auto fall = [&, &guard = guard, &s = s] {
      if (i + 1 < insns.size())
        pending[i + 1].emplace_back(guard, s);
    };
    if (
      it->is_skip() || it->type == LOCATION || it->is_assume() ||
      it->type == DEAD)
      fall();
    else if (it->type == DECL)
    {
      s.erase(to_code_decl2t(it->code).value);
      fall();
    }
    else if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (!is_symbol2t(a.target) || touches_memory_or_sideeffect(a.source))
        return false;
      expr2tc v = read_in(a.source, s);
      if (is_nil_expr(v))
        return false;
      s[to_symbol2t(a.target).thename] = v;
      fall();
    }
    else if (it->is_goto())
    {
      if (it->is_backwards_goto() || it->targets.size() != 1)
        return false;
      auto t = index.find(&*it->targets.front());
      if (t == index.end() || t->second <= i)
        return false;
      if (is_true(it->guard))
      {
        pending[t->second].emplace_back(guard, s);
        continue;
      }
      if (touches_memory_or_sideeffect(it->guard))
        return false;
      expr2tc c = read_in(it->guard, s);
      if (is_nil_expr(c))
        return false;
      pending[t->second].emplace_back(and2tc(guard, c), s);
      if (i + 1 < insns.size())
        pending[i + 1].emplace_back(and2tc(guard, not2tc(c)), s);
    }
    else if (it->is_return())
    {
      const expr2tc &op = to_code_return2t(it->code).operand;
      if (!is_nil_expr(op))
      {
        expr2tc v = read_in(op, s);
        if (is_nil_expr(v))
          return false;
        s["#return"] = v;
      }
      effect.returns.emplace_back(guard, s);
    }
    else if (it->is_end_function())
      effect.returns.emplace_back(guard, s);
    else if (it->is_function_call())
    {
      const code_function_call2t &call = to_code_function_call2t(it->code);
      if (!is_symbol2t(call.function))
        return false;
      const irep_idt &name = to_symbol2t(call.function).thename;
      if (name == "c:@F@exit" || name == "c:@F@abort")
        continue; // the path ends
      auto callee = goto_functions.function_map.find(name);
      if (
        callee == goto_functions.function_map.end() ||
        !is_code_type(callee->second.type))
        return false;
      const code_type2t &ct = to_code_type(callee->second.type);
      if (ct.argument_names.size() != call.operands.size())
        return false;
      statet callee_entry = s;
      for (size_t k = 0; k < call.operands.size(); ++k)
      {
        expr2tc v = read_in(call.operands[k], s);
        if (is_nil_expr(v))
          return false;
        callee_entry[ct.argument_names[k]] = v;
      }
      call_effectt inner;
      if (!run_once(
            goto_functions, callee->second, callee_entry, depth + 1, inner))
        return false;
      if (inner.returns.empty())
        continue; // every path of the call ends the program
      for (auto &[g, after] : inner.returns)
      {
        g = and2tc(guard, g);
        if (!is_nil_expr(call.ret))
        {
          if (!is_symbol2t(call.ret) || !after.count("#return"))
            return false;
          after[to_symbol2t(call.ret).thename] = after["#return"];
        }
        after.erase("#return");
      }
      auto [g, after] = join(inner.returns);
      if (i + 1 < insns.size())
        pending[i + 1].emplace_back(g, after);
    }
    else
      return false;
  }
  return true;
}

/// Run the loop from the initial state for up to @p bound iterations, each
/// with a fresh valid input, and ask whether a state repeats before a call
/// to exit or abort ends the run. The globals are all the state the loop
/// carries, so a repeated state is a lasso the inputs can follow forever.
/// Unlike the period-1 check no state is left free: every state on the lasso
/// is reached from the initial one.
bool reaches_cycle_from_init(
  const goto_functionst &goto_functions,
  const irep_idt &callee,
  const expr2tc &input_arg,
  const std::vector<BigInt> &valid_inputs,
  const std::vector<expr2tc> &result_writes,
  const std::unordered_map<irep_idt, BigInt, irep_id_hash> &initial_state,
  unsigned bound,
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
  const auto fn = goto_functions.function_map.find(callee);
  if (!is_code_type(fn->second.type))
    return false;
  const code_type2t &ct = to_code_type(fn->second.type);
  if (ct.argument_names.size() != 1)
    return false;
  // The loop stores the call's result; a global there would be state the
  // step does not update.
  for (const expr2tc &w : result_writes)
    if (s0.count(to_symbol2t(w).thename))
      return false;

  expr2tc formula = gen_true_expr();
  expr2tc alive_so_far = gen_true_expr();
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

    statet entry = states[t];
    entry[ct.argument_names[0]] = in;
    call_effectt effect;
    if (!run_once(goto_functions, fn->second, entry, 0, effect))
      return false;
    if (effect.returns.empty())
      return false;
    expr2tc alive = gen_false_expr();
    for (const auto &[g, after] : effect.returns)
      alive = or2tc(alive, g);
    alive_so_far = and2tc(alive_so_far, alive);

    // Name the next state, so that each step reads symbols, not the
    // whole history.
    statet next;
    for (const auto &[name, v0] : s0)
    {
      expr2tc value = states[t].at(name);
      for (const auto &[g, after] : effect.returns)
      {
        auto a = after.find(name);
        if (a == after.end())
          return false;
        value = if2tc(v0->type, g, a->second, value);
      }
      next[name] = symbol2tc(
        v0->type,
        "termination::lasso::" + id2string(name) + "::" + i2string(t + 1));
      formula = and2tc(formula, equality2tc(next[name], value));
    }
    states.push_back(next);

    expr2tc same_as_earlier = gen_false_expr();
    for (unsigned i = 0; i <= t; ++i)
    {
      expr2tc same = gen_true_expr();
      for (const auto &[name, v] : next)
        same = and2tc(same, equality2tc(states[i].at(name), v));
      same_as_earlier = or2tc(same_as_earlier, same);
    }
    repeats = or2tc(repeats, and2tc(alive_so_far, same_as_earlier));
  }
  return is_sat(and2tc(formula, repeats), options, ns);
}

/// No instruction before the loop head writes a global or calls anything,
/// so the loop starts in the state __ESBMC_main initialised. A validator's
/// check of a bare cycle waypoint reads no state.
bool loop_starts_at_initial_state(
  const goto_programt &body,
  goto_programt::const_targett head,
  const namespacet &ns)
{
  for (auto it = body.instructions.begin(); it != head; ++it)
  {
    if (
      it->is_function_call() &&
      !(is_symbol2t(to_code_function_call2t(it->code).function) &&
        to_symbol2t(to_code_function_call2t(it->code).function).thename ==
          "c:@F@__ESBMC_witness_assume"))
      return false;
    if (it->is_backwards_goto())
      return false;
    if (!it->is_assign())
      continue;
    const expr2tc &target = to_code_assign2t(it->code).target;
    if (!is_symbol2t(target))
      return false;
    const symbolt *sym = ns.lookup(to_symbol2t(target).thename);
    if (!sym || sym->static_lifetime)
      return false;
  }
  return true;
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
  // MVP scope: only the eca-rers2012 finite-state-machine shape. Each
  // expected-false eca benchmark wraps an input-validation while(1)
  // around a single call to `calculate_output(input)`; if there's a
  // state s and an input i ∈ valid_inputs such that none of the
  // branches in calculate_output fire, then the body falls through to
  // its trailing `return -2` with state unchanged, and the demon can
  // pick that (s, i) forever. We discharge the existence question via
  // a single SMT SAT query — no symex required, just the IF guards
  // harvested from the goto program, with the callee's formal input
  // substituted by the caller's actual input and globals constrained
  // to their initialisers harvested from `__ESBMC_main`.
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
      // Identify the callee's single formal parameter so we can
      // substitute its symbol with the caller's actual input.
      irep_idt callee_formal_name;
      if (is_code_type(callee_it->second.type))
      {
        const code_type2t &ct = to_code_type(callee_it->second.type);
        if (ct.argument_names.size() == 1)
          callee_formal_name = ct.argument_names[0];
      }
      std::vector<expr2tc> branch_conds;
      bool flat = collect_callee_branch_guards(callee_it->second, branch_conds);
      // A call result stored where the guards read it changes the state the
      // fall-through left alone, so the next iteration differs.
      std::unordered_map<irep_idt, expr2tc, irep_id_hash> guard_reads;
      for (const expr2tc &c : branch_conds)
        collect_free_symbols(c, guard_reads);
      if (
        std::any_of(
          result_writes.begin(), result_writes.end(), [&](const expr2tc &w) {
            return guard_reads.count(to_symbol2t(w).thename);
          }))
        flat = false;
      std::unordered_set<irep_idt, irep_id_hash> callee_writes;
      collect_writes(callee_it->second, callee_writes);
      if (
        (flat && check_period_1_fixpoint(
                   input_arg,
                   callee_formal_name,
                   valid_inputs,
                   branch_conds,
                   initial_state,
                   callee_writes,
                   options,
                   ns)) ||
        (loop_starts_at_initial_state(
           f_it->second.body, loop.get_original_loop_head(), ns) &&
         reaches_cycle_from_init(
           goto_functions,
           callee_name,
           input_arg,
           valid_inputs,
           result_writes,
           initial_state,
           24,
           options,
           ns)))
      {
        loop_location = head;
        return tvt(tvt::TV_FALSE);
      }
    }
  }
  return tvt(tvt::TV_UNKNOWN);
}
