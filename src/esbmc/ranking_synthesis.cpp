#include <esbmc/ranking_synthesis.h>

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

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace
{
/// A recognized loop in the supported shape:
///   loop_head:  IF !(a REL b) GOTO exit     (REL is a relational compare)
///   <straight-line body of assigns>
///   GOTO loop_head
/// We record the guard relation operands and the body's assignment
/// effects so the ranking obligations can be built.
struct assignt
{
  expr2tc lhs;
  expr2tc rhs;
};

struct ranking_loopt
{
  // The loop-head IF instruction (guard is its (negated) condition).
  goto_programt::targett head;
  // The back-edge GOTO.
  goto_programt::targett back;
  // Guard, expressed positively (loop continues while this holds):
  // i.e. the negation of the IF guard. A relational compare a REL b.
  expr2tc guard;
  // Straight-line body assignments, in program order.
  std::vector<assignt> body;
};

/// Is @p e a relational comparison (>, >=, <, <=) — the guard shape we
/// can derive a difference measure from?
bool is_relational(const expr2tc &e)
{
  return is_greaterthan2t(e) || is_greaterthanequal2t(e) ||
         is_lessthan2t(e) || is_lessthanequal2t(e);
}

/// True if @p e contains a memory-dependent subexpression (dereference,
/// array index, struct/union member, byte op). Such expressions can't
/// be handed to the solver directly — they need symex's dereferencing
/// and memory model to be resolved first. We only build obligations
/// from scalar (symbol/constant/arithmetic) expressions, so loops whose
/// guard or transition touch memory are out of scope for this pass and
/// must fall back to UNKNOWN rather than crash the solver.
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

/// Try to recognize the supported loop shape. Fills @p out and returns
/// true on success. Returns false (caller treats as UNKNOWN) for any
/// shape we don't handle yet:
///   - loop_head must be `IF !(a REL b) GOTO exit` with a relational
///     guard,
///   - the body (instructions strictly between head and back-edge) must
///     be straight-line ASSIGNs only — no further GOTO/IF, no
///     FUNCTION_CALL, no nested back-edge,
///   - the back-edge must be an unconditional GOTO to the head.
bool recognize_loop(const loopst &loop, ranking_loopt &out)
{
  out.head = loop.get_original_loop_head();
  out.back = loop.get_original_loop_exit();

  // Head must be the loop-condition IF: a forward GOTO whose guard is
  // the *negated* loop condition.
  if (!out.head->is_goto())
    return false;
  if (is_nil_expr(out.head->guard) || is_true(out.head->guard))
    return false;

  // Positive loop guard = !(head guard). Must be a relational compare
  // over scalar (non-memory) operands — a bare dereference/index/member
  // can't be converted to SMT without symex's memory model, so we leave
  // those loops to the existing machinery.
  expr2tc pos_guard = out.head->guard;
  make_not(pos_guard);
  simplify(pos_guard);
  if (!is_relational(pos_guard) || touches_memory(pos_guard))
    return false;
  out.guard = pos_guard;

  // Body: instructions strictly between head and back-edge. Only
  // straight-line assigns over scalar lvalues allowed.
  out.body.clear();
  for (auto it = std::next(out.head); it != out.back; ++it)
  {
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (touches_memory(a.target) || touches_memory(a.source))
        return false;
      out.body.push_back({a.target, a.source});
      continue;
    }
    // Skippable bookkeeping that doesn't affect the transition relation.
    if (it->is_skip() || it->is_location() || it->type == DEAD)
      continue;
    // Anything else (GOTO, IF, FUNCTION_CALL, ASSUME/ASSERT, nested
    // back-edge, ...) takes us outside the supported shape.
    return false;
  }

  // Back-edge must be an unconditional GOTO (the `GOTO loop_head`).
  if (!out.back->is_goto() || !out.back->is_backwards_goto())
    return false;

  return true;
}

/// Substitute every occurrence of @p from with @p to inside @p e
/// (structural value equality on expr2tc). Used to compute m' = m after
/// the loop body's transition: replace each assigned lhs by its rhs.
expr2tc subst(const expr2tc &e, const expr2tc &from, const expr2tc &to)
{
  if (is_nil_expr(e))
    return e;
  if (e == from)
    return to;
  expr2tc r = e;
  r->Foreach_operand([&](expr2tc &op) { op = subst(op, from, to); });
  return r;
}

/// Apply the loop body's assignments, in order, to expression @p e —
/// yielding e evaluated in the post-iteration state. Each assignment
/// lhs := rhs rewrites later references; we substitute sequentially so
/// `x = x + 1; y = x` composes correctly.
expr2tc apply_body(const expr2tc &e, const std::vector<assignt> &body)
{
  expr2tc r = e;
  for (const auto &a : body)
    r = subst(r, a.lhs, a.rhs);
  return r;
}

/// True iff `formula` is UNSATISFIABLE under a fresh solver. A false
/// return (SAT / error / unknown) means the obligation is not
/// discharged.
bool is_unsat(const expr2tc &formula, optionst &options, const namespacet &ns)
{
  std::unique_ptr<smt_convt> solver(create_solver("", ns, options));
  solver->assert_expr(formula);
  return solver->dec_solve() == smt_convt::P_UNSATISFIABLE;
}

bool prove_loop_terminates(
  const ranking_loopt &rl,
  optionst &options,
  const namespacet &ns)
{
  // Derive the candidate measure m and its guard-implied lower-bound
  // slack from the relational guard. For `a > b`, m = a - b and the
  // guard implies m >= 1; for `a >= b`, m >= 0. For `<`/`<=` the roles
  // of the two sides swap. The measure is computed in a widened type so
  // the subtraction cannot overflow (signed overflow would be UB and
  // make the proof unsound).
  expr2tc a, b;
  BigInt bound; // guard => m >= bound
  if (is_greaterthan2t(rl.guard))
  {
    a = to_greaterthan2t(rl.guard).side_1;
    b = to_greaterthan2t(rl.guard).side_2;
    bound = 1;
  }
  else if (is_greaterthanequal2t(rl.guard))
  {
    a = to_greaterthanequal2t(rl.guard).side_1;
    b = to_greaterthanequal2t(rl.guard).side_2;
    bound = 0;
  }
  else if (is_lessthan2t(rl.guard))
  {
    a = to_lessthan2t(rl.guard).side_2;
    b = to_lessthan2t(rl.guard).side_1;
    bound = 1;
  }
  else if (is_lessthanequal2t(rl.guard))
  {
    a = to_lessthanequal2t(rl.guard).side_2;
    b = to_lessthanequal2t(rl.guard).side_1;
    bound = 0;
  }
  else
    return false;

  // Only integer-typed operands: the difference measure and its
  // bitvector widening are defined for integers, not floats/pointers.
  if (!is_bv_type(a->type) || !is_bv_type(b->type))
    return false;

  // m = widen(a) - widen(b), in a type wide enough that the subtraction
  // of the operand types cannot overflow.
  type2tc wide = get_int_type(64);
  expr2tc m = sub2tc(wide, typecast2tc(wide, a), typecast2tc(wide, b));

  // m' = m after one loop body iteration.
  expr2tc m_prime = apply_body(m, rl.body);

  expr2tc L = constant_int2tc(wide, bound);

  // Bounded-below obligation: guard ∧ (m < L) must be UNSAT, i.e. the
  // guard implies m >= L (so the measure cannot decrease past the
  // floor without the guard becoming false → loop exits).
  expr2tc bound_violated = and2tc(rl.guard, lessthan2tc(m, L));
  if (!is_unsat(bound_violated, options, ns))
    return false;

  // Decrease obligation: guard ∧ ¬(m' < m) must be UNSAT, i.e. under
  // the guard the measure strictly decreases every iteration.
  expr2tc not_decreasing =
    and2tc(rl.guard, greaterthanequal2tc(m_prime, m));
  if (!is_unsat(not_decreasing, options, ns))
    return false;

  log_debug(
    "termination",
    "ranking function proved loop terminates: measure={}",
    from_expr(ns, "", m));
  return true;
}

/// True if the program contains recursion — any cycle in the static
/// call graph (direct or mutual). The ranking check reasons only about
/// LOOP termination; a recursive call is a second, independent source
/// of non-termination it does not account for. Without this guard, a
/// purely-recursive non-terminating program (no loops at all) would be
/// declared terminating vacuously — an unsound wrong-true (e.g.
/// termination-crafted/RecursiveNonterminating, MutualRecursion,
/// recursified_*, ll_*_rec). If recursion is present we must return
/// UNKNOWN and let the existing machinery decide.
bool has_recursion(const goto_functionst &goto_functions)
{
  // Build call edges: f -> set of directly-called function names.
  // Skip body.hide library helpers: they are verification scaffolding
  // (e.g. __ESBMC_atexit_handler, which makes an indirect call through
  // stdlib_atexit_key and is linked into EVERY program) and are handled
  // by the existing machinery, not the ranking analysis. Counting their
  // function-pointer calls as recursion hazards would disable the check
  // for every benchmark.
  std::unordered_map<irep_idt, std::vector<irep_idt>, irep_id_hash> callees;
  for (const auto &fn : goto_functions.function_map)
  {
    if (!fn.second.body_available || fn.second.body.hide)
      continue;
    auto &outs = callees[fn.first];
    for (const auto &ins : fn.second.body.instructions)
    {
      if (!ins.is_function_call())
        continue;
      const code_function_call2t &call = to_code_function_call2t(ins.code);
      if (is_symbol2t(call.function))
        outs.push_back(to_symbol2t(call.function).thename);
      else
        // Function-pointer call in user code: target unknown, can't
        // rule out a cycle. Treat as a recursion hazard (conservative).
        return true;
    }
  }

  // DFS for a back-edge (cycle). colors: 0=unvisited, 1=on-stack, 2=done.
  std::unordered_map<irep_idt, int, irep_id_hash> color;
  std::function<bool(const irep_idt &)> dfs = [&](const irep_idt &f) -> bool {
    color[f] = 1;
    auto it = callees.find(f);
    if (it != callees.end())
    {
      for (const auto &g : it->second)
      {
        int c = color.count(g) ? color[g] : 0;
        if (c == 1) // back-edge to a function on the current stack
          return true;
        if (c == 0 && callees.count(g) && dfs(g))
          return true;
      }
    }
    color[f] = 2;
    return false;
  };

  for (const auto &fn : callees)
    if (color.count(fn.first) == 0 && dfs(fn.first))
      return true;
  return false;
}
} // namespace

tvt try_prove_termination_by_ranking(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns)
{
  // Recursion is a non-termination source the loop-ranking analysis does
  // not model. If the program has any call-graph cycle, we cannot claim
  // termination from loop ranks alone — bail to UNKNOWN.
  if (has_recursion(goto_functions))
    return tvt(tvt::TV_UNKNOWN);

  // Every loop in every function with a body must be proven terminating
  // for the program to be declared terminating. A single loop we cannot
  // handle makes the whole check inconclusive.
  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;
    if (f_it->second.body.hide)
      continue; // library helpers: handled by the existing machinery

    goto_loopst loops(f_it->first, goto_functions, f_it->second);
    for (const auto &loop : loops.get_loops())
    {
      ranking_loopt rl;
      if (!recognize_loop(loop, rl))
        return tvt(tvt::TV_UNKNOWN);
      if (!prove_loop_terminates(rl, options, ns))
        return tvt(tvt::TV_UNKNOWN);
    }
  }

  // All loops proven terminating. With recursion ruled out above and
  // every natural loop ranked, no non-termination source remains
  // (vacuously true if there are no loops).
  return tvt(tvt::TV_TRUE);
}
