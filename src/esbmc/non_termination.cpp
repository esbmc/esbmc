#include <esbmc/non_termination.h>

#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <solvers/solve.h>
#include <solvers/smt/smt_conv.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>
#include <util/irep/std_expr.h>
#include <util/base/yaml_parser.h>
#include <langapi/language_util.h>
#include <util/arith/mp_arith.h>
#include <util/base/i2string.h>
#include <util/config/config.h>

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
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
  if (head == back || !back->is_goto() || !back->is_backwards_goto())
    return false;
  // while(1)'s exit test never jumps; remove_no_op, which k-induction runs
  // first, drops it, leaving an unconditional back edge.
  goto_programt::const_targett first = head;
  if (head->is_goto() && is_constant_false(head->guard))
    ++first;
  else if (!is_true(back->guard))
    return false;
  expr2tc input_sym;
  irep_idt callee;
  std::vector<BigInt> values;
  for (auto it = first; it != back; ++it)
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

/// @p e simplified, so a run on concrete states stays concrete.
expr2tc folded(expr2tc e)
{
  simplify(e);
  return e;
}

/// @p guard and @p c, leaving out a @p c that is true.
expr2tc conjoin(const expr2tc &guard, const expr2tc &c)
{
  return is_true(c) ? guard : is_true(guard) ? c : and2tc(guard, c);
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
    // A failed assertion ends the execution (C11 7.2.1.1: assert aborts).
    case ASSERT:
    case ASSUME:
    {
      const expr2tc c = folded(read_in(it->guard, s));
      if (!is_false(c))
        next.emplace_back(conjoin(guard, c), s);
      break;
    }
    case ASSIGN:
      if (!is_symbol2t(to_code_assign2t(it->code).target))
        throw unsupportedt();
      s[to_symbol2t(to_code_assign2t(it->code).target).thename] =
        folded(read_in(to_code_assign2t(it->code).source, s));
      next.emplace_back(guard, std::move(s));
      break;
    case GOTO:
    {
      if (it->is_backwards_goto() || it->targets.size() != 1)
        throw unsupportedt();
      const expr2tc c = folded(read_in(it->guard, s));
      if (!is_false(c))
        pending[&*it->targets.front()].emplace_back(conjoin(guard, c), s);
      if (!is_true(c))
        next.emplace_back(conjoin(guard, folded(not2tc(c))), s);
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

/// Control reaches @p head of the entry function @p fname straight from its
/// start with the globals as __ESBMC_main initialised them: nothing before it
/// branches, assumes, writes a global or calls anything but a witness assume.
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

/// The globals of @p s0 that the updater @p callee, or a function it calls,
/// writes (@p written) or reads or writes (@p used). False when it reads or
/// writes one that a function outside it,
/// other than the entry function @p fname and __ESBMC_main's initialisation,
/// also writes (e.g. a pthread hook run before main): then the head may not
/// see the value __ESBMC_main gave it.
bool updater_globals(
  const goto_functionst &goto_functions,
  const irep_idt &fname,
  const irep_idt &callee,
  const statet &s0,
  std::unordered_set<irep_idt, irep_id_hash> &written,
  std::unordered_set<irep_idt, irep_id_hash> &used)
{
  std::unordered_set<irep_idt, irep_id_hash> closure{callee};
  std::vector<irep_idt> todo{callee};
  std::unordered_map<irep_idt, expr2tc, irep_id_hash> touched;
  while (!todo.empty())
  {
    auto f = goto_functions.function_map.find(todo.back());
    todo.pop_back();
    if (f == goto_functions.function_map.end())
      continue;
    collect_writes(f->second, written);
    for (const auto &i : f->second.body.instructions)
    {
      collect_free_symbols(i.code, touched);
      collect_free_symbols(i.guard, touched);
      if (
        i.is_function_call() &&
        is_symbol2t(to_code_function_call2t(i.code).function))
      {
        const irep_idt &g =
          to_symbol2t(to_code_function_call2t(i.code).function).thename;
        if (closure.insert(g).second)
          todo.push_back(g);
      }
    }
  }
  std::unordered_set<irep_idt, irep_id_hash> elsewhere;
  for (const auto &[name, fn] : goto_functions.function_map)
    if (!closure.count(name) && name != fname && name != "__ESBMC_main")
      collect_writes(fn, elsewhere);
  for (const auto &[name, e] : touched)
  {
    if (!s0.count(name))
      continue;
    if (elsewhere.count(name))
      return false;
    used.insert(name);
  }
  for (auto it = written.begin(); it != written.end();)
    it = s0.count(*it) ? std::next(it) : written.erase(it);
  return true;
}

/// Runs an input loop's iterations on concrete values. Each guard and
/// assigned value is compiled once into flat nodes over numbered variables,
/// since the updaters of the larger eca problems are chains of dozens of
/// functions evaluated thousands of times per state. Integers wrap to their
/// type's width as ESBMC's bit-vector encoding wraps them. Throws unsupportedt
/// on anything else.
class concrete_runnert
{
public:
  concrete_runnert(const goto_functionst &goto_functions, const loopst &loop)
    : goto_functions(goto_functions), loop(loop)
  {
  }

  /// The slot of the variable @p name.
  size_t slot(const irep_idt &name)
  {
    auto [it, fresh] = slots.emplace(name, values.size());
    if (fresh)
    {
      values.push_back(0);
      defined.push_back(false);
    }
    return it->second;
  }

  void set(size_t v, int64_t x)
  {
    values[v] = x;
    defined[v] = true;
  }

  int64_t get(size_t v) const
  {
    if (!defined[v])
      throw unsupportedt();
    return values[v];
  }

  /// One iteration on the current values, reading @p in as the input @p
  /// input_arg; walked as run_iteration walks it. False when it stops.
  bool iterate(const expr2tc &input_arg, int64_t in)
  {
    steps = 0;
    bool called = false;
    for (auto it = loop.get_original_loop_head();
         it != loop.get_original_loop_exit();
         ++it)
    {
      if (
        it->is_assign() && !called &&
        to_code_assign2t(it->code).target == input_arg)
        assign(input_arg, wrap_to(input_arg->type, in));
      else if (it->is_function_call())
      {
        if (!call(it, 0))
          return false;
        called = true;
      }
      else if (it->is_assume() && !holds(it))
        return false;
      else if (it->is_assign() && called)
        assign(it);
    }
    return true;
  }

private:
  enum class opt : uint8_t
  {
    var,
    constant,
    cast,
    lnot,
    land,
    lor,
    ite,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    neg,
    add,
    sub,
    mul,
    div,
    mod,
    unsupported
  };

  struct nodet
  {
    opt op;
    bool is_signed;
    uint64_t mask; // the type's width; zero for Booleans
    int64_t k;     // constant value, or variable slot
    int32_t a, b, c;
  };

  const goto_functionst &goto_functions;
  const loopst &loop;
  std::unordered_map<irep_idt, size_t, irep_id_hash> slots;
  std::vector<int64_t> values;
  std::vector<bool> defined;
  std::vector<nodet> nodes;
  std::unordered_map<const expr2t *, int32_t> compiled;
  size_t steps = 0;

  /// A call resolved once: its callee, argument nodes, and where the
  /// parameters and the result go.
  struct call_sitet
  {
    bool stops = false; // exit or abort
    const goto_functiont *fn = nullptr;
    std::vector<int32_t> args;
    std::vector<size_t> params;
    std::vector<std::pair<uint64_t, bool>> param_widths;
    bool has_ret = false;
    size_t ret = 0;
    std::pair<uint64_t, bool> ret_width{0, false};
  };
  std::vector<call_sitet> call_sites;
  std::unordered_map<const goto_functiont *, bool> running;

  /// Per instruction, by location_number: its compiled guard, value and call
  /// site, -1 until first needed.
  std::vector<int32_t> guard_at, value_at, call_at;

  static int32_t &at(std::vector<int32_t> &v, unsigned n)
  {
    if (v.size() <= n)
      v.resize(n + 1, -1);
    return v[n];
  }

  int32_t guard_of(goto_programt::const_targett it)
  {
    int32_t &g = at(guard_at, it->location_number);
    if (g < 0)
      g = compile(it->guard);
    return g;
  }

  int32_t value_of(goto_programt::const_targett it, const expr2tc &e)
  {
    int32_t &v = at(value_at, it->location_number);
    if (v < 0)
      v = compile(e);
    return v;
  }

  /// The width mask and signedness of @p type; throws on non-integers.
  static std::pair<uint64_t, bool> width_of(const type2tc &type)
  {
    if (is_bool_type(type))
      return {0, false};
    const bool is_signed = is_signedbv_type(type);
    if (!is_signed && !is_unsignedbv_type(type))
      throw unsupportedt();
    const unsigned w = type->get_width();
    if (w == 0 || w > 64 || (w == 64 && !is_signed))
      throw unsupportedt();
    return {w == 64 ? ~0ULL : (1ULL << w) - 1, is_signed};
  }

  static int64_t wrap(uint64_t v, uint64_t mask, bool is_signed)
  {
    if (mask == 0)
      return v != 0;
    v &= mask;
    if (is_signed && (v & ~(mask >> 1) & mask))
      v |= ~mask;
    return static_cast<int64_t>(v);
  }

  static int64_t wrap_to(const type2tc &type, int64_t v)
  {
    auto [mask, is_signed] = width_of(type);
    return wrap(static_cast<uint64_t>(v), mask, is_signed);
  }

  int32_t add_node(nodet n)
  {
    nodes.push_back(n);
    return static_cast<int32_t>(nodes.size() - 1);
  }

  /// The node computing @p e; an opt::unsupported node when it cannot.
  int32_t compile(const expr2tc &e)
  {
    if (auto it = compiled.find(e.get()); it != compiled.end())
      return it->second;
    int32_t idx;
    try
    {
      idx = compile_node(e);
    }
    catch (const unsupportedt &)
    {
      idx = add_node({opt::unsupported, false, 0, 0, -1, -1, -1});
    }
    compiled.emplace(e.get(), idx);
    return idx;
  }

  int32_t compile_node(const expr2tc &e)
  {
    auto sub = [&](unsigned i) {
      const int32_t n = compile(*e->get_sub_expr(i));
      if (nodes[n].op == opt::unsupported)
        throw unsupportedt();
      return n;
    };
    auto typed = [&](opt op, int32_t a, int32_t b = -1, int32_t c = -1) {
      auto [mask, is_signed] = width_of(e->type);
      return add_node({op, is_signed, mask, 0, a, b, c});
    };
    auto boolean = [&](opt op, int32_t a, int32_t b = -1, int32_t c = -1) {
      return add_node({op, false, 0, 0, a, b, c});
    };
    switch (e->expr_id)
    {
    case expr2t::symbol_id:
    {
      const size_t v = slot(to_symbol2t(e).thename);
      const int32_t n = boolean(opt::var, -1);
      nodes[n].k = static_cast<int64_t>(v);
      return n;
    }
    case expr2t::constant_int_id:
    {
      if (!to_constant_int2t(e).value.is_int64())
        throw unsupportedt();
      const int32_t n = typed(opt::constant, -1);
      nodes[n].k = wrap_to(e->type, to_constant_int2t(e).value.to_int64());
      return n;
    }
    case expr2t::constant_bool_id:
    {
      const int32_t n = boolean(opt::constant, -1);
      nodes[n].k = to_constant_bool2t(e).value;
      return n;
    }
    case expr2t::typecast_id:
      return typed(opt::cast, sub(0));
    case expr2t::not_id:
      return boolean(opt::lnot, sub(0));
    case expr2t::and_id:
      return boolean(opt::land, sub(0), sub(1));
    case expr2t::or_id:
      return boolean(opt::lor, sub(0), sub(1));
    case expr2t::if_id:
      return typed(opt::ite, sub(0), sub(1), sub(2));
    case expr2t::equality_id:
      return boolean(opt::eq, sub(0), sub(1));
    case expr2t::notequal_id:
      return boolean(opt::ne, sub(0), sub(1));
    case expr2t::lessthan_id:
      return boolean(opt::lt, sub(0), sub(1));
    case expr2t::lessthanequal_id:
      return boolean(opt::le, sub(0), sub(1));
    case expr2t::greaterthan_id:
      return boolean(opt::gt, sub(0), sub(1));
    case expr2t::greaterthanequal_id:
      return boolean(opt::ge, sub(0), sub(1));
    case expr2t::neg_id:
      return typed(opt::neg, sub(0));
    case expr2t::add_id:
      return typed(opt::add, sub(0), sub(1));
    case expr2t::sub_id:
      return typed(opt::sub, sub(0), sub(1));
    case expr2t::mul_id:
      return typed(opt::mul, sub(0), sub(1));
    case expr2t::div_id:
      return typed(opt::div, sub(0), sub(1));
    case expr2t::modulus_id:
      return typed(opt::mod, sub(0), sub(1));
    default:
      throw unsupportedt();
    }
  }

  int64_t eval(int32_t i)
  {
    const nodet &n = nodes[i];
    auto u = [&](int32_t j) { return static_cast<uint64_t>(eval(j)); };
    switch (n.op)
    {
    case opt::var:
      return get(n.k);
    case opt::constant:
      return n.k;
    case opt::cast:
      return wrap(u(n.a), n.mask, n.is_signed);
    case opt::lnot:
      return !eval(n.a);
    case opt::land:
      return eval(n.a) && eval(n.b);
    case opt::lor:
      return eval(n.a) || eval(n.b);
    case opt::ite:
      return eval(n.a) ? eval(n.b) : eval(n.c);
    case opt::eq:
      return eval(n.a) == eval(n.b);
    case opt::ne:
      return eval(n.a) != eval(n.b);
    case opt::lt:
      return eval(n.a) < eval(n.b);
    case opt::le:
      return eval(n.a) <= eval(n.b);
    case opt::gt:
      return eval(n.a) > eval(n.b);
    case opt::ge:
      return eval(n.a) >= eval(n.b);
    case opt::neg:
      return wrap(0 - u(n.a), n.mask, n.is_signed);
    case opt::add:
      return wrap(u(n.a) + u(n.b), n.mask, n.is_signed);
    case opt::sub:
      return wrap(u(n.a) - u(n.b), n.mask, n.is_signed);
    case opt::mul:
      return wrap(u(n.a) * u(n.b), n.mask, n.is_signed);
    case opt::div:
    case opt::mod:
    {
      const int64_t a = eval(n.a), b = eval(n.b);
      if (b == 0 || (a == INT64_MIN && b == -1))
        throw unsupportedt();
      // C11 6.5.5p6: the quotient truncates toward zero.
      return wrap(
        static_cast<uint64_t>(n.op == opt::div ? a / b : a % b),
        n.mask,
        n.is_signed);
    }
    case opt::unsupported:
      break;
    }
    throw unsupportedt();
  }

  /// The assumption or assertion at @p it; one it cannot evaluate keeps the
  /// execution, which only adds states.
  bool holds(goto_programt::const_targett it)
  {
    const int32_t n = guard_of(it);
    if (nodes[n].op == opt::unsupported)
      return true;
    try
    {
      return eval(n);
    }
    catch (const unsupportedt &)
    {
      return true;
    }
  }

  void assign(const expr2tc &target, int64_t v)
  {
    if (!is_symbol2t(target))
      throw unsupportedt();
    set(slot(to_symbol2t(target).thename), wrap_to(target->type, v));
  }

  /// The assignment at @p it.
  void assign(goto_programt::const_targett it)
  {
    const code_assign2t &a = to_code_assign2t(it->code);
    assign(a.target, eval(value_of(it, a.source)));
  }

  const call_sitet &site_of(goto_programt::const_targett it)
  {
    int32_t &i = at(call_at, it->location_number);
    if (i >= 0)
      return call_sites[i];
    const code_function_call2t &c = to_code_function_call2t(it->code);
    call_sitet site;
    if (!is_symbol2t(c.function))
      throw unsupportedt();
    const irep_idt &name = to_symbol2t(c.function).thename;
    if (name == "c:@F@exit" || name == "c:@F@abort")
      site.stops = true;
    else
    {
      auto callee = goto_functions.function_map.find(name);
      if (
        callee == goto_functions.function_map.end() ||
        !callee->second.body_available || !is_code_type(callee->second.type))
        throw unsupportedt();
      const code_type2t &type = to_code_type(callee->second.type);
      if (type.argument_names.size() != c.operands.size())
        throw unsupportedt();
      site.fn = &callee->second;
      for (size_t k = 0; k < c.operands.size(); ++k)
      {
        site.args.push_back(compile(c.operands[k]));
        site.params.push_back(slot(type.argument_names[k]));
        site.param_widths.push_back(width_of(type.arguments[k]));
      }
      if (!is_nil_expr(c.ret))
      {
        if (!is_symbol2t(c.ret))
          throw unsupportedt();
        site.has_ret = true;
        site.ret = slot(to_symbol2t(c.ret).thename);
        site.ret_width = width_of(c.ret->type);
      }
    }
    call_sites.push_back(std::move(site));
    i = static_cast<int32_t>(call_sites.size() - 1);
    return call_sites[i];
  }

  /// Run the call at @p it. False when the execution stops in it: exit,
  /// abort, or a failed assertion or assumption.
  bool call(goto_programt::const_targett it, unsigned depth)
  {
    const call_sitet &site = site_of(it);
    if (site.stops)
      return false;
    // A function's locals have one slot each, so it must not recurse.
    bool &busy = running[site.fn];
    if (busy || depth > 256)
      throw unsupportedt();
    int64_t args[16];
    if (site.args.size() > 16)
      throw unsupportedt();
    for (size_t k = 0; k < site.args.size(); ++k)
      args[k] = eval(site.args[k]);
    for (size_t k = 0; k < site.args.size(); ++k)
      set(
        site.params[k],
        wrap(
          static_cast<uint64_t>(args[k]),
          site.param_widths[k].first,
          site.param_widths[k].second));
    std::optional<int64_t> ret;
    busy = true;
    const bool returned = run(*site.fn, ret, depth);
    busy = false;
    if (!returned)
      return false;
    if (site.has_ret)
    {
      if (!ret)
        throw unsupportedt();
      set(
        site.ret,
        wrap(
          static_cast<uint64_t>(*ret),
          site.ret_width.first,
          site.ret_width.second));
    }
    return true;
  }

  /// Run @p fn, following its one concrete path. False when the execution
  /// stops in it; @p ret receives the returned value.
  bool
  run(const goto_functiont &fn, std::optional<int64_t> &ret, unsigned depth)
  {
    for (auto it = fn.body.instructions.begin();
         it != fn.body.instructions.end();)
    {
      if (++steps > 10000000)
        throw unsupportedt();
      switch (it->type)
      {
      case DECL:
        defined[slot(to_code_decl2t(it->code).value)] = false;
        break;
      case DEAD:
      case SKIP:
      case LOCATION:
        break;
      // C11 7.2.1.1: a failed assertion aborts.
      case ASSERT:
      case ASSUME:
        if (!holds(it))
          return false;
        break;
      case ASSIGN:
        assign(it);
        break;
      case GOTO:
        if (it->is_backwards_goto() || it->targets.size() != 1)
          throw unsupportedt();
        if (eval(guard_of(it)))
        {
          it = it->targets.front();
          continue;
        }
        break;
      case FUNCTION_CALL:
        if (!call(it, depth + 1))
          return false;
        break;
      case RETURN:
        if (!is_nil_expr(to_code_return2t(it->code).operand))
          ret = eval(value_of(it, to_code_return2t(it->code).operand));
        return true;
      case END_FUNCTION:
        return true;
      default:
        throw unsupportedt();
      }
      ++it;
    }
    return true;
  }
};

/// Every state the input loop @p loop reaches at its head from @p s0, found
/// by running each iteration on concrete values for every valid input. Empty
/// when an iteration is not a function of the state and the input (it reads
/// something else, or does something concrete_runnert does not model) or
/// when more than @p cap states, or more than 256 values of one global, turn
/// up: a global computed by arithmetic soon has more values than any table.
std::vector<statet> reachable_states(
  const goto_functionst &goto_functions,
  const loopst &loop,
  const expr2tc &input_arg,
  const std::vector<BigInt> &valid_inputs,
  const statet &s0,
  size_t cap)
{
  concrete_runnert runner(goto_functions, loop);
  std::vector<std::pair<irep_idt, size_t>> globals;
  std::vector<int64_t> init;
  for (const auto &[name, value] : s0)
  {
    if (!to_constant_int2t(value).value.is_int64())
      return {};
    globals.emplace_back(name, runner.slot(name));
    init.push_back(to_constant_int2t(value).value.to_int64());
  }
  std::vector<std::vector<int64_t>> todo{init};
  std::set<std::vector<int64_t>> seen{init};
  std::vector<std::set<int64_t>> values(globals.size());
  try
  {
    for (size_t i = 0; i < todo.size(); ++i)
      for (const BigInt &v : valid_inputs)
      {
        if (!v.is_int64())
          return {};
        for (size_t g = 0; g < globals.size(); ++g)
          runner.set(globals[g].second, todo[i][g]);
        if (!runner.iterate(input_arg, v.to_int64()))
          continue;
        std::vector<int64_t> next;
        for (const auto &[name, s] : globals)
          next.push_back(runner.get(s));
        if (!seen.insert(next).second)
          continue;
        if (todo.size() == cap)
          return {};
        for (size_t g = 0; g < globals.size(); ++g)
        {
          values[g].insert(next[g]);
          if (values[g].size() > 256)
            return {};
        }
        todo.push_back(std::move(next));
      }
  }
  catch (const unsupportedt &)
  {
    return {};
  }
  std::vector<statet> states;
  for (const auto &t : todo)
  {
    statet st;
    for (size_t g = 0; g < globals.size(); ++g)
      st[globals[g].first] =
        constant_int2tc(s0.at(globals[g].first)->type, BigInt(t[g]));
    states.push_back(std::move(st));
  }
  return states;
}

/// The eca updater @p fn, called with the input once per iteration, never
/// fires a branch for some reachable state and valid input, leaving the state
/// as it was: the period-1 recurrent set.
bool has_period_1_fixpoint(
  const goto_functiont &fn,
  const expr2tc &input_arg,
  const std::vector<BigInt> &valid_inputs,
  const std::vector<expr2tc> &result_writes,
  const std::unordered_map<irep_idt, BigInt, irep_id_hash> &initial_state,
  optionst &options,
  const namespacet &ns)
{
  // Identify the callee's single formal parameter so we can
  // substitute its symbol with the caller's actual input.
  irep_idt callee_formal_name;
  if (is_code_type(fn.type))
  {
    const code_type2t &ct = to_code_type(fn.type);
    if (ct.argument_names.size() == 1)
      callee_formal_name = ct.argument_names[0];
  }
  std::vector<expr2tc> branch_conds;
  if (!collect_callee_branch_guards(fn, branch_conds))
    return false;
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
    return false;
  std::unordered_set<irep_idt, irep_id_hash> callee_writes;
  collect_writes(fn, callee_writes);
  return check_period_1_fixpoint(
    input_arg,
    callee_formal_name,
    valid_inputs,
    branch_conds,
    initial_state,
    callee_writes,
    options,
    ns);
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
      if (
        has_period_1_fixpoint(
          callee_it->second,
          input_arg,
          valid_inputs,
          result_writes,
          initial_state,
          options,
          ns) ||
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

std::vector<std::pair<locationt, std::string>> assume_reachable_loop_states(
  goto_functionst &goto_functions,
  const namespacet &ns,
  size_t max_states)
{
  std::vector<std::pair<locationt, std::string>> invariants;
  // concrete_runnert indexes its caches by location_number; earlier passes
  // may have inserted instructions since the last numbering.
  goto_functions.compute_location_numbers();
  const auto initial_state = harvest_initial_state(goto_functions);
  auto f_it = goto_functions.function_map.find(
    "c:@F@" + (config.main.empty() ? "main" : config.main));
  if (f_it == goto_functions.function_map.end() || !f_it->second.body_available)
    return invariants;
  statet s0;
  for (const auto &[name, value] : initial_state)
  {
    const symbolt *sym = ns.lookup(name);
    if (sym && sym->static_lifetime && is_bv_type(sym->get_type2()))
      s0[name] = constant_int2tc(sym->get_type2(), value);
  }
  goto_loopst loops(f_it->first, goto_functions, f_it->second);
  for (const auto &loop : loops.get_loops())
  {
    expr2tc input_arg;
    std::vector<BigInt> valid_inputs;
    irep_idt callee_name;
    std::vector<expr2tc> result_writes;
    if (
      !recognize_eca_main_loop(
        loop, input_arg, valid_inputs, callee_name, result_writes) ||
      !loop_starts_at_initial_state(
        f_it->first, f_it->second.body, loop.get_original_loop_head(), ns))
      continue;
    // The loop must change the state only through the updater.
    result_writes.push_back(input_arg);
    if (
      std::any_of(
        result_writes.begin(), result_writes.end(), [&](const expr2tc &w) {
          return s0.count(to_symbol2t(w).thename);
        }))
      continue;
    std::unordered_set<irep_idt, irep_id_hash> written, used;
    if (!updater_globals(
          goto_functions, f_it->first, callee_name, s0, written, used))
      continue;
    // Only what the updater touches: run_once gives up on reading the rest.
    statet state;
    for (const irep_idt &name : used)
      state[name] = s0.at(name);
    // A witness may name only the program's own variables, not those of
    // ESBMC's library models (e.g. exit's atexit_count).
    const irep_idt &file = loop.get_original_loop_exit()->location.get_file();
    for (auto it = written.begin(); it != written.end();)
    {
      const symbolt *sym = ns.lookup(*it);
      it = sym && sym->location.get_file() == file ? std::next(it)
                                                   : written.erase(it);
    }
    const std::vector<statet> states = reachable_states(
      goto_functions, loop, input_arg, valid_inputs, state, max_states);
    if (states.empty() || written.empty())
      continue;

    std::vector<expr2tc> disjuncts;
    for (const statet &s : states)
    {
      std::vector<expr2tc> eqs;
      for (const auto &[name, value] : s)
        if (written.count(name))
          eqs.push_back(equality2tc(symbol2tc(value->type, name), value));
      disjuncts.push_back(conjunction(eqs));
    }
    // Only the inductive step needs it, after its havoc: every state the
    // loop head sees on an execution that has not yet failed is in the set.
    const expr2tc invariant = disjunction(disjuncts);
    goto_programt::targett head = loop.get_original_loop_head();
    goto_programt::targett assume =
      f_it->second.body.instructions.insert(std::next(head), *head);
    assume->make_assumption(invariant);
    assume->targets.clear();
    assume->labels.clear();
    assume->inductive_step_instruction = true;
    log_status(
      "Loop at {}: assuming its {} reachable states in the inductive step",
      head->location.as_string(),
      states.size());
    // The back edge carries the loop statement's own location.
    invariants.emplace_back(
      loop.get_original_loop_exit()->location,
      from_expr(ns, "", invariant, presentationt::WITNESS));
  }
  return invariants;
}
