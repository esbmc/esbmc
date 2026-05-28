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

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
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
  // The body's set of straight-line execution paths. A non-branching body
  // produces exactly one path (the assignment sequence). A body with a
  // single `if (cond) ... else ...` produces two paths, one per arm. The
  // ranking obligations and invariant inductiveness must hold on EVERY
  // path; failure on any path falls back to UNKNOWN.
  std::vector<std::vector<assignt>> paths;
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

/// Collect straight-line assigns in [@p first, @p last). Returns false if
/// any instruction is anything other than DECL/ASSIGN/skip/location/DEAD,
/// or a memory-touching assign — the caller treats that as an unrecognized
/// body shape. ASSIGNs are appended to @p out in program order.
bool collect_straight_line(
  goto_programt::const_targett first,
  goto_programt::const_targett last,
  std::vector<assignt> &out)
{
  for (auto it = first; it != last; ++it)
  {
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (touches_memory(a.target) || touches_memory(a.source))
        return false;
      out.push_back({a.target, a.source});
      continue;
    }
    if (
      it->is_skip() || it->is_location() || it->type == DEAD ||
      it->type == DECL)
      continue;
    return false;
  }
  return true;
}

/// Try to recognize the supported loop shape. Fills @p out and returns
/// true on success. Returns false (caller treats as UNKNOWN) for any
/// shape we don't handle yet. Two body shapes are accepted:
///
///   (A) Straight-line. The instructions strictly between head and back-
///       edge are scalar ASSIGNs only (one path).
///
///   (B) A single in-body `if/else` of straight-line arms. The instruction
///       sequence between head and back-edge looks like
///         <pre-assigns>
///         IF !cond GOTO else_label
///         <then-assigns>
///         GOTO merge_label
///         else_label:
///         <else-assigns>
///         merge_label:
///         <post-assigns>
///       (the `else` arm may be empty: then `IF !cond GOTO merge_label`
///       and there is no inner GOTO/else_label). We yield two paths,
///       <pre> ++ <then> ++ <post> and <pre> ++ <else> ++ <post>; the
///       ranking obligations and invariant inductiveness must hold on
///       both.
///
/// In all cases the loop head must be `IF !(a REL b) GOTO exit` with a
/// relational guard over scalar BV operands, and the back-edge must be an
/// unconditional GOTO to the head.
bool recognize_loop(const loopst &loop, ranking_loopt &out)
{
  out.head = loop.get_original_loop_head();
  out.back = loop.get_original_loop_exit();

  if (!out.head->is_goto())
    return false;
  if (is_nil_expr(out.head->guard) || is_true(out.head->guard))
    return false;

  expr2tc pos_guard = out.head->guard;
  make_not(pos_guard);
  simplify(pos_guard);
  if (!is_relational(pos_guard) || touches_memory(pos_guard))
    return false;
  out.guard = pos_guard;

  if (!out.back->is_goto() || !out.back->is_backwards_goto())
    return false;

  out.paths.clear();

  // Shape (A): straight-line body — no GOTO/IF instructions between head
  // and back-edge.
  auto body_begin = std::next(out.head);
  bool has_internal_goto = false;
  for (auto it = body_begin; it != out.back; ++it)
    if (it->is_goto())
    {
      has_internal_goto = true;
      break;
    }
  if (!has_internal_goto)
  {
    std::vector<assignt> path;
    if (!collect_straight_line(body_begin, out.back, path))
      return false;
    out.paths.push_back(std::move(path));
    return true;
  }

  // Shape (B): a single in-body if/else. Walk pre-assigns, find a
  // conditional forward IF into the body, optionally split with an
  // unconditional GOTO to the merge target.
  std::vector<assignt> pre;
  auto it = body_begin;
  while (it != out.back && !it->is_goto())
  {
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (touches_memory(a.target) || touches_memory(a.source))
        return false;
      pre.push_back({a.target, a.source});
    }
    else if (
      !it->is_skip() && !it->is_location() && it->type != DEAD &&
      it->type != DECL)
      return false;
    ++it;
  }
  if (it == out.back)
    return false; // a GOTO must exist by has_internal_goto

  // The first GOTO must be a conditional forward IF (the if-condition).
  if (
    !it->is_goto() || is_true(it->guard) || it->is_backwards_goto() ||
    it->targets.size() != 1)
    return false;
  auto if_target = it->targets.front();
  if (if_target->location_number <= it->location_number)
    return false; // not forward
  expr2tc if_cond = it->guard; // raw IF guard (the negated then-condition)
  if (touches_memory(if_cond))
    return false;
  auto then_begin = std::next(it);

  // Find the unconditional GOTO that ends the then-arm. If we don't see
  // one before the if_target, the else-arm is empty and the merge IS
  // if_target. Otherwise the GOTO's target is the merge_label.
  auto then_end = then_begin;
  while (then_end != if_target && !then_end->is_goto())
    ++then_end;
  goto_programt::const_targett else_begin, merge_label;
  if (then_end == if_target)
  {
    // No else branch: the IF jumps straight to the merge.
    else_begin = if_target; // empty else
    merge_label = if_target;
  }
  else
  {
    // then_end is the closing GOTO of the then-arm; its target is the
    // merge label. The instructions [if_target .. merge_label) are the
    // else-arm.
    if (
      !then_end->is_goto() || !is_true(then_end->guard) ||
      then_end->is_backwards_goto() || then_end->targets.size() != 1)
      return false;
    merge_label = then_end->targets.front();
    if (merge_label->location_number <= then_end->location_number)
      return false;
    else_begin = if_target;
  }
  if (merge_label == out.back || merge_label->location_number >
                                   out.back->location_number)
    return false; // merge must fall before the back-edge

  // Collect the three straight-line spans.
  std::vector<assignt> then_arm, else_arm, post;
  if (!collect_straight_line(then_begin, then_end, then_arm))
    return false;
  if (then_end != if_target)
  {
    // else_arm = [if_target .. merge_label)
    if (!collect_straight_line(if_target, merge_label, else_arm))
      return false;
  }
  if (!collect_straight_line(merge_label, out.back, post))
    return false;

  // Build the two paths: pre + then + post, and pre + else + post.
  // The IF's raw guard `if_cond` is the NEGATION of the C-source `then`
  // condition: `if (X) S` lowers to `IF !X GOTO else_label; S; ...`. So
  // the THEN-arm runs when `!if_cond` holds and the ELSE-arm runs when
  // `if_cond` holds. We don't currently use these path conditions in the
  // obligations (we require decrease on every path unconditionally), but
  // they could refine the invariant-inductiveness check in a future step.
  auto build = [&](const std::vector<assignt> &mid)
  {
    std::vector<assignt> p = pre;
    p.insert(p.end(), mid.begin(), mid.end());
    p.insert(p.end(), post.begin(), post.end());
    return p;
  };
  out.paths.push_back(build(then_arm));
  out.paths.push_back(build(else_arm));
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

/// Simultaneously replace every variable that is a key of @p post by its
/// mapped expression inside @p e, in a SINGLE traversal: a replacement is
/// never re-scanned for further substitution. This gives parallel-
/// assignment semantics, unlike repeated single substitutions which would
/// let one mapping cascade into another.
expr2tc subst_parallel(
  const expr2tc &e,
  const std::map<expr2tc, expr2tc> &post)
{
  if (is_nil_expr(e))
    return e;
  auto it = post.find(e);
  if (it != post.end())
    return it->second;
  expr2tc r = e;
  r->Foreach_operand([&](expr2tc &op) { op = subst_parallel(op, post); });
  return r;
}

/// Recompute a scalar integer expression in the int64 domain: cast every
/// leaf (symbol / constant) to int64 and rebuild each arithmetic node so
/// the operation itself is performed in 64 bits. With operands at most
/// 32-bit, sums/differences of a few terms stay well inside int64, so this
/// removes the modular-bitvector wraparound that a 32-bit add (e.g.
/// d1 = d2 + 1 at INT_MAX) would otherwise introduce when reasoning about
/// invariant bounds. Returns nil if @p e contains a non-arithmetic node we
/// can't faithfully widen (e.g. NONDET sideeffects, function-call results,
/// or any other opaque construct), so the caller drops the atom rather
/// than feed an unsound or solver-stalling expression to the solver.
expr2tc widen_arith(const expr2tc &e)
{
  if (is_nil_expr(e))
    return expr2tc();
  type2tc wide = get_int_type(64);
  if (is_symbol2t(e) || is_constant_int2t(e))
    return typecast2tc(wide, e);
  if (is_typecast2t(e))
    return widen_arith(to_typecast2t(e).from);
  auto rec2 = [&](const expr2tc &a, const expr2tc &b) -> std::pair<expr2tc, expr2tc>
  {
    return {widen_arith(a), widen_arith(b)};
  };
  if (is_add2t(e))
  {
    auto [l, r] = rec2(to_add2t(e).side_1, to_add2t(e).side_2);
    return (is_nil_expr(l) || is_nil_expr(r)) ? expr2tc() : add2tc(wide, l, r);
  }
  if (is_sub2t(e))
  {
    auto [l, r] = rec2(to_sub2t(e).side_1, to_sub2t(e).side_2);
    return (is_nil_expr(l) || is_nil_expr(r)) ? expr2tc() : sub2tc(wide, l, r);
  }
  if (is_mul2t(e))
  {
    auto [l, r] = rec2(to_mul2t(e).side_1, to_mul2t(e).side_2);
    return (is_nil_expr(l) || is_nil_expr(r)) ? expr2tc() : mul2tc(wide, l, r);
  }
  if (is_neg2t(e))
  {
    expr2tc v = widen_arith(to_neg2t(e).value);
    return is_nil_expr(v) ? expr2tc() : neg2tc(wide, v);
  }
  // Anything else (NONDET sideeffect, function call, dereference, etc.):
  // we cannot faithfully widen, so signal "skip this atom".
  return expr2tc();
}

/// Build the loop body's transition as a parallel substitution and apply
/// it to @p e, yielding e in the post-iteration state. Each assignment
/// lhs := rhs is evaluated against the PRE-state: we process assignments
/// in program order, resolving each rhs through the post-state values
/// computed so far (so `t = x; y = t` sees t's pre-iteration definition),
/// then substitute all resulting lhs ↦ value pairs into @p e at once.
/// This is faithful to simultaneous body semantics, whereas substituting
/// each assignment into @p e in sequence would let a later rhs read an
/// earlier assignment's already-rewritten value — an unfaithful
/// transition (e.g. `a = a - b; b = a` would collapse to `a - a`).
expr2tc apply_body(const expr2tc &e, const std::vector<assignt> &body)
{
  // post[v] = value of v after the body, expressed in pre-state terms.
  std::map<expr2tc, expr2tc> post;
  for (const auto &a : body)
    post[a.lhs] = subst_parallel(a.rhs, post);
  return subst_parallel(e, post);
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

/// Derive a candidate difference measure m and its guard-implied lower
/// bound L from a relational guard `a REL b`. For `a > b`, m = a - b
/// and the guard implies m >= 1; for `a >= b`, m >= 0; `<`/`<=` swap
/// the operands. The measure is computed in a type wide enough that the
/// subtraction of the operand types cannot overflow — signed overflow
/// would be UB and make the proof unsound. Returns false if the guard
/// is not relational or its operands are not integer-typed (the
/// difference/widening is only defined for integers).
bool measure_from_guard(const expr2tc &guard, expr2tc &m, expr2tc &L)
{
  expr2tc a, b;
  BigInt bound;
  if (is_greaterthan2t(guard))
  {
    a = to_greaterthan2t(guard).side_1;
    b = to_greaterthan2t(guard).side_2;
    bound = 1;
  }
  else if (is_greaterthanequal2t(guard))
  {
    a = to_greaterthanequal2t(guard).side_1;
    b = to_greaterthanequal2t(guard).side_2;
    bound = 0;
  }
  else if (is_lessthan2t(guard))
  {
    a = to_lessthan2t(guard).side_2;
    b = to_lessthan2t(guard).side_1;
    bound = 1;
  }
  else if (is_lessthanequal2t(guard))
  {
    a = to_lessthanequal2t(guard).side_2;
    b = to_lessthanequal2t(guard).side_1;
    bound = 0;
  }
  else
    return false;

  if (!is_bv_type(a->type) || !is_bv_type(b->type))
    return false;

  // The measure is m = (int64)a - (int64)b. Value-preserving extension to
  // int64 keeps each operand in its source range, so the subtraction cannot
  // overflow as long as both fit comfortably below int64: 32-bit operands
  // give a difference in [-(2^32-1), 2^32-1], far inside int64. A 64-bit
  // operand, however, can produce a difference outside int64, which would
  // wrap under modular bitvector subtraction and let a non-decreasing or
  // unbounded measure spuriously satisfy the obligations. Refuse those.
  if (a->type->get_width() > 32 || b->type->get_width() > 32)
    return false;

  type2tc wide = get_int_type(64);
  m = sub2tc(wide, typecast2tc(wide, a), typecast2tc(wide, b));
  L = constant_int2tc(wide, bound);
  return true;
}

/// Conjoin a list of atoms into a single expression (true if empty).
expr2tc conjoin(const std::vector<expr2tc> &atoms)
{
  if (atoms.empty())
    return gen_true_expr();
  expr2tc r = atoms.front();
  for (size_t i = 1; i < atoms.size(); ++i)
    r = and2tc(r, atoms[i]);
  return r;
}

/// Collect the symbol names referenced (free) in @p e.
void collect_symbols(const expr2tc &e, std::set<irep_idt> &out)
{
  if (is_nil_expr(e))
    return;
  if (is_symbol2t(e))
  {
    out.insert(to_symbol2t(e).thename);
    return;
  }
  e->foreach_operand([&](const expr2tc &op) { collect_symbols(op, out); });
}

/// Flatten a conjunction `a && b && ...` into its individual atoms,
/// dropping `true` operands. Anything not an `and` is returned as a single
/// atom (the caller decides whether it's usable).
void flatten_and(const expr2tc &e, std::vector<expr2tc> &out)
{
  if (is_nil_expr(e) || is_true(e))
    return;
  if (is_and2t(e))
  {
    flatten_and(to_and2t(e).side_1, out);
    flatten_and(to_and2t(e).side_2, out);
    return;
  }
  out.push_back(e);
}

/// Decompose a relational atom into one or two one-sided atoms suitable
/// for the synthesizer's atom pool: `a == b` ↦ {a ≥ b, a ≤ b}; the four
/// inequalities pass through. Returns false if the atom isn't a usable
/// relational shape (or touches memory).
bool decompose_relational(const expr2tc &e, std::vector<expr2tc> &out)
{
  if (is_nil_expr(e) || touches_memory(e))
    return false;
  if (is_relational(e))
  {
    out.push_back(e);
    return true;
  }
  if (is_equality2t(e))
  {
    const equality2t &eq = to_equality2t(e);
    if (!is_bv_type(eq.side_1->type) || !is_bv_type(eq.side_2->type))
      return false;
    out.push_back(greaterthanequal2tc(eq.side_1, eq.side_2));
    out.push_back(lessthanequal2tc(eq.side_1, eq.side_2));
    return true;
  }
  return false;
}

/// Seeds collected from the loop's pre-header, used as base-case facts.
/// `constants[v]` is the value of the LAST `ASSIGN v = <int constant>`
/// before the head from a dominating straight-line position; `path_atoms`
/// are extra relational atoms that hold at loop entry because they come
/// from `IF guard GOTO past_loop` instructions whose fall-through edge
/// (which leads to the loop) implies `¬guard`, and whose free variables
/// are not reassigned between the IF and the loop head.
struct prefix_seedst
{
  std::map<irep_idt, BigInt> constants;
  std::vector<expr2tc> path_atoms;
};

/// Walk forward from @p start (a candidate branch of an IF) following the
/// single-successor edges that the prefix scanner accepts, and report
/// whether the walk reaches @p head. The walk follows fall-through on
/// straight-line instructions, follows unconditional GOTOs, and traverses
/// conditional IFs by exploring both successors (bounded by the prefix
/// span). Stops on RETURN/END_FUNCTION (the path exits) or on hitting
/// @p head (success). Returns false on anything else (FUNCTION_CALL,
/// memory-touching assign, back-edge, multi-target GOTO) — we treat those
/// the same as exits since we wouldn't accept them anyway. The check is
/// purely structural; we use it only to decide which IF branch carries
/// the seed condition into the loop.
bool reaches_head(
  goto_programt::const_targett start,
  goto_programt::const_targett head,
  goto_programt::const_targett end_it)
{
  std::set<unsigned> visited;
  std::vector<goto_programt::const_targett> stack;
  stack.push_back(start);
  while (!stack.empty())
  {
    auto it = stack.back();
    stack.pop_back();
    if (it == end_it)
      continue;
    if (!visited.insert(it->location_number).second)
      continue;
    if (it == head)
      return true;
    if (it->is_skip() || it->is_location() || it->type == DECL)
    {
      stack.push_back(std::next(it));
      continue;
    }
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (touches_memory(a.target) || touches_memory(a.source))
        continue;
      stack.push_back(std::next(it));
      continue;
    }
    if (it->is_goto() && !it->is_backwards_goto() && it->targets.size() == 1)
    {
      stack.push_back(it->targets.front());
      if (!is_true(it->guard))
        stack.push_back(std::next(it));
      continue;
    }
    // RETURN/END_FUNCTION/FUNCTION_CALL/anything else: path exits.
  }
  return false;
}

/// Collect seeds from the loop's pre-header. The base case of an assumed
/// invariant requires every seed to hold on EVERY first entry to the loop
/// head. We enforce this with a forward sweep from function entry to the
/// head, allowing the following:
///
///  * DECL / ASSIGN / skip / location instructions (an assign updates the
///    constant-seed map and invalidates any path atom mentioning the
///    target's name).
///  * Conditional IFs: for each IF we use a bounded forward reachability
///    check from each branch (target and fall-through) to determine which
///    one carries control to the loop head. If exactly one branch reaches
///    the head, the corresponding condition (the IF's guard for the jump
///    branch, its negation for the fall-through branch) holds on every
///    path to the head and contributes a path atom. If neither branch
///    reaches the head we bail (the loop must be unreachable from here);
///    if both reach we get no path atom from this IF (the conditions on
///    the two branches are disjunctive, so no useful fact holds on every
///    path through them).
///  * Targets that are reached only via a previously-traversed IF jump
///    (i.e., we have already promoted that IF's seed): the target marks
///    the merge point of the IF, control through it satisfies the seed.
///
/// Anything else (FUNCTION_CALL, back-edge, multi-target GOTO, memory-
/// touching assign, jump targets reachable from an unanalysed edge) makes
/// the prefix unsafe for syntactic seeding, and we return empty seeds so
/// the loop falls back to the bare ranking check.
prefix_seedst collect_seeds(
  const goto_programt &body,
  goto_programt::const_targett head)
{
  prefix_seedst seeds;
  // Targets we've justified by an earlier accepted IF jump (so meeting one
  // of them mid-sweep is not a soundness break).
  std::set<unsigned> justified_targets;
  // Pending IF-derived atoms: free-vars set so a later assign to one of
  // those vars can invalidate the atom before it reaches the head.
  struct pending_atom
  {
    expr2tc atom;
    std::set<irep_idt> free_vars;
  };
  std::vector<pending_atom> pending;
  auto end_it = body.instructions.end();

  // We walk a single live path from entry to the loop head, advancing one
  // instruction at a time except at conditional IFs where we may have to
  // jump to the target if the fall-through doesn't lead to the head. The
  // sweep terminates when @p it reaches @p head.
  for (auto it = body.instructions.begin(); it != head;)
  {
    if (it->is_target() && !justified_targets.count(it->location_number))
      return {};
    if (it->is_skip() || it->is_location() || it->type == DECL)
    {
      ++it;
      continue;
    }
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (touches_memory(a.target) || touches_memory(a.source))
        return {};
      if (!is_symbol2t(a.target))
        return {};
      const irep_idt &name = to_symbol2t(a.target).thename;
      if (is_constant_int2t(a.source))
        seeds.constants[name] = to_constant_int2t(a.source).value;
      else
        seeds.constants.erase(name);
      pending.erase(
        std::remove_if(
          pending.begin(),
          pending.end(),
          [&](const pending_atom &p) { return p.free_vars.count(name); }),
        pending.end());
      ++it;
      continue;
    }
    if (it->is_goto() && !is_true(it->guard) && !it->is_backwards_goto() &&
        it->targets.size() == 1)
    {
      auto tgt = it->targets.front();
      // Determine which branch carries control to the loop head.
      bool fall_reaches = reaches_head(std::next(it), head, end_it);
      bool jump_reaches = reaches_head(tgt, head, end_it);
      if (!fall_reaches && !jump_reaches)
        return {}; // loop head unreachable through this IF: bail
      expr2tc seed_cond;
      auto next = std::next(it);
      if (fall_reaches && !jump_reaches)
      {
        // Only fall-through leads to the loop ⇒ ¬guard holds at the head.
        // Continue the sweep at the next sequential instruction.
        seed_cond = it->guard;
        make_not(seed_cond);
        simplify(seed_cond);
      }
      else if (jump_reaches && !fall_reaches)
      {
        // Only the jump leads to the loop ⇒ guard holds at the head.
        // Skip the fall-through (it does not reach the head) and resume
        // the sweep at the jump target, which is justified by this IF.
        seed_cond = it->guard;
        simplify(seed_cond);
        justified_targets.insert(tgt->location_number);
        next = tgt;
      }
      else
      {
        // Both branches reach the head: no useful seed (disjunctive). The
        // sweep must cover both arms, but a linear walk can't — bail to
        // stay sound. (Such shapes are rare and handled by future work.)
        return {};
      }
      std::vector<expr2tc> conjuncts;
      flatten_and(seed_cond, conjuncts);
      for (const expr2tc &c : conjuncts)
      {
        std::vector<expr2tc> atoms;
        if (!decompose_relational(c, atoms))
          continue;
        for (const expr2tc &a : atoms)
        {
          std::set<irep_idt> fv;
          collect_symbols(a, fv);
          if (fv.empty())
            continue;
          pending.push_back({a, std::move(fv)});
        }
      }
      it = next;
      continue;
    }
    // RETURN / END_FUNCTION / FUNCTION_CALL / multi-target GOTO / back-edge:
    // we cannot continue safely.
    return {};
  }
  for (auto &p : pending)
    seeds.path_atoms.push_back(std::move(p.atom));
  return seeds;
}

/// Synthesize a supporting loop invariant as a conjunction of one-sided
/// constant bound atoms (v >= c / v <= c) over body-modified variables.
/// Each atom is seeded from a constant pre-header assignment (so it holds
/// on loop entry — the base case) and kept only if it is inductive: under
/// the current candidate conjunction and the guard, the body preserves it.
/// We reach a fixpoint by repeatedly dropping non-preserved atoms (atoms
/// are only ever removed, so this terminates); the survivors form an
/// inductive invariant that is sound to assume in the ranking obligations.
expr2tc synthesize_invariant(
  const ranking_loopt &rl,
  const goto_programt &body,
  optionst &options,
  const namespacet &ns)
{
  prefix_seedst seeds = collect_seeds(body, rl.head);
  if (seeds.constants.empty() && seeds.path_atoms.empty())
    return gen_true_expr();

  // Build candidate atoms over the int64 domain so the body's arithmetic on
  // these bounds cannot wrap: a 32-bit increment like d1 = d2 + 1 wraps
  // under modular bitvector semantics at INT_MAX, which would spuriously
  // break an otherwise-inductive bound. Operands are at most 32-bit
  // (checked), so widened values stay well inside int64. Each atom is
  // tracked alongside its post-state image (the same atom with body-
  // modified vars rewritten to their post-iteration value), so the
  // inductiveness check reasons entirely in the non-wrapping wide domain.
  struct cand
  {
    expr2tc atom;                 // pre-state atom, holds at entry
    std::vector<expr2tc> primes;  // post-state image per body path
  };
  // Build one post-substitution map per body path: each `post[i][v]` is the
  // value of `v` after path `i`, expressed in pre-state terms. The atom
  // inductiveness check then needs the atom preserved by EVERY path.
  std::vector<std::map<expr2tc, expr2tc>> path_posts(rl.paths.size());
  for (size_t i = 0; i < rl.paths.size(); ++i)
    for (const auto &a : rl.paths[i])
      path_posts[i][a.lhs] = subst_parallel(a.rhs, path_posts[i]);

  auto post_state_of = [&](const expr2tc &e, size_t i)
  { return widen_arith(subst_parallel(e, path_posts[i])); };

  // Body-modified scalars are the union of lhs across all paths.
  std::map<irep_idt, expr2tc> body_vars; // name -> exemplar lhs (for type)
  for (const auto &path : rl.paths)
    for (const auto &a : path)
      if (
        is_symbol2t(a.lhs) && is_bv_type(a.lhs->type) &&
        a.lhs->type->get_width() <= 32)
        body_vars.emplace(to_symbol2t(a.lhs).thename, a.lhs);

  std::vector<cand> atoms;
  // Constant-init seeds: for each body-modified scalar var that has a
  // constant pre-loop assignment c, add both v >= c and v <= c. The
  // fixpoint will drop whichever direction any path doesn't preserve.
  for (const auto &kv : body_vars)
  {
    auto s = seeds.constants.find(kv.first);
    if (s == seeds.constants.end())
      continue;
    type2tc wide = get_int_type(64);
    expr2tc c = constant_int2tc(wide, s->second);
    expr2tc v = widen_arith(kv.second);
    if (is_nil_expr(v))
      continue;
    // Build per-path widened post-state values; skip the atom if any path
    // cannot be faithfully widened (NONDET body assignment, opaque expr).
    std::vector<expr2tc> vps;
    vps.reserve(rl.paths.size());
    bool ok = true;
    for (size_t i = 0; i < rl.paths.size(); ++i)
    {
      expr2tc vp = post_state_of(kv.second, i);
      if (is_nil_expr(vp))
      {
        ok = false;
        break;
      }
      vps.push_back(vp);
    }
    if (!ok)
      continue;
    cand ge, le;
    ge.atom = greaterthanequal2tc(v, c);
    le.atom = lessthanequal2tc(v, c);
    for (const auto &vp : vps)
    {
      ge.primes.push_back(greaterthanequal2tc(vp, c));
      le.primes.push_back(lessthanequal2tc(vp, c));
    }
    atoms.push_back(std::move(ge));
    atoms.push_back(std::move(le));
  }
  // Path-condition seeds: each atom comes from `IF guard GOTO past_loop`
  // whose fall-through edge (reaching the head) implies `¬guard`, with the
  // free variables proven not to be reassigned between the IF and the
  // head. The atom holds at entry by construction; its post-state image
  // gets the same widened-substitution treatment.
  for (const expr2tc &a : seeds.path_atoms)
  {
    // Rebuild the atom with both sides widened, so the comparison happens
    // in int64 (avoids spurious wrap when the post-state side has +1 etc).
    if (!is_relational(a))
      continue;
    expr2tc lhs, rhs;
    if (is_greaterthan2t(a))
    {
      lhs = to_greaterthan2t(a).side_1;
      rhs = to_greaterthan2t(a).side_2;
    }
    else if (is_greaterthanequal2t(a))
    {
      lhs = to_greaterthanequal2t(a).side_1;
      rhs = to_greaterthanequal2t(a).side_2;
    }
    else if (is_lessthan2t(a))
    {
      lhs = to_lessthan2t(a).side_1;
      rhs = to_lessthan2t(a).side_2;
    }
    else
    {
      lhs = to_lessthanequal2t(a).side_1;
      rhs = to_lessthanequal2t(a).side_2;
    }
    if (
      !is_bv_type(lhs->type) || !is_bv_type(rhs->type) ||
      lhs->type->get_width() > 32 || rhs->type->get_width() > 32)
      continue;
    expr2tc l = widen_arith(lhs), r = widen_arith(rhs);
    if (is_nil_expr(l) || is_nil_expr(r))
      continue;
    // Build per-path widened post-state images; skip the atom if any path
    // cannot be faithfully widened.
    std::vector<std::pair<expr2tc, expr2tc>> lr_per_path;
    lr_per_path.reserve(rl.paths.size());
    bool ok = true;
    for (size_t i = 0; i < rl.paths.size(); ++i)
    {
      expr2tc lp = post_state_of(lhs, i);
      expr2tc rp = post_state_of(rhs, i);
      if (is_nil_expr(lp) || is_nil_expr(rp))
      {
        ok = false;
        break;
      }
      lr_per_path.emplace_back(lp, rp);
    }
    if (!ok)
      continue;
    auto make_atom = [&](const expr2tc &x, const expr2tc &y) -> expr2tc
    {
      if (is_greaterthan2t(a))
        return greaterthan2tc(x, y);
      if (is_greaterthanequal2t(a))
        return greaterthanequal2tc(x, y);
      if (is_lessthan2t(a))
        return lessthan2tc(x, y);
      return lessthanequal2tc(x, y);
    };
    cand c;
    c.atom = make_atom(l, r);
    for (const auto &lr : lr_per_path)
      c.primes.push_back(make_atom(lr.first, lr.second));
    atoms.push_back(std::move(c));
  }
  if (atoms.empty())
    return gen_true_expr();

  // Fixpoint: drop atoms not preserved by SOME body path under the guard
  // and the other surviving atoms. An atom A survives iff for EVERY path,
  //   (∧ atoms) ∧ guard ∧ ¬A'_path   is UNSAT,
  // where A'_path is the atom's post-state image on that path. A single
  // failing path drops the atom. Atoms are only ever removed, so this
  // terminates.
  bool changed = true;
  while (changed && !atoms.empty())
  {
    changed = false;
    std::vector<expr2tc> cur;
    for (const auto &a : atoms)
      cur.push_back(a.atom);
    expr2tc inv = conjoin(cur);
    for (size_t i = 0; i < atoms.size(); ++i)
    {
      bool preserved = true;
      for (const expr2tc &prime : atoms[i].primes)
      {
        expr2tc neg = prime;
        make_not(neg);
        if (!is_unsat(and2tc(and2tc(inv, rl.guard), neg), options, ns))
        {
          preserved = false;
          break;
        }
      }
      if (!preserved)
      {
        atoms.erase(atoms.begin() + i);
        changed = true;
        break; // restart with the weakened conjunction
      }
    }
  }

  std::vector<expr2tc> survivors;
  for (const auto &a : atoms)
    survivors.push_back(a.atom);
  return conjoin(survivors);
}

bool prove_loop_terminates(
  const ranking_loopt &rl,
  const goto_programt &body,
  optionst &options,
  const namespacet &ns)
{
  // Candidate measure m and guard-implied lower bound L from the guard.
  expr2tc m, L;
  if (!measure_from_guard(rl.guard, m, L))
    return false;

  // Supporting invariant: an inductive conjunction of constant bounds that
  // holds on entry. Sound to assume in the obligations below (it is true
  // every iteration). For loops that need no extra facts this is just true.
  expr2tc inv = synthesize_invariant(rl, body, options, ns);

  // Bounded-below obligation: inv ∧ guard ∧ (m < L) must be UNSAT, i.e.
  // under the invariant the guard implies m >= L (so the measure cannot
  // decrease past the floor without the guard becoming false → loop exits).
  // This obligation doesn't depend on the body, so it's checked once.
  if (!is_unsat(and2tc(and2tc(inv, rl.guard), lessthan2tc(m, L)), options, ns))
    return false;

  // Decrease obligation: under the invariant and the guard, the measure
  // strictly decreases on EVERY body path. For each path we build m'
  // separately (via apply_body over that path's assignments) and require
  // inv ∧ guard ∧ ¬(m' < m) UNSAT. If any path fails to decrease, the
  // loop is not provably terminating by this rank.
  for (const auto &path : rl.paths)
  {
    expr2tc m_prime = apply_body(m, path);
    if (!is_unsat(
          and2tc(and2tc(inv, rl.guard), greaterthanequal2tc(m_prime, m)),
          options,
          ns))
      return false;
  }

  log_debug(
    "termination",
    "ranking function proved loop terminates: measure={} invariant={}",
    from_expr(ns, "", m),
    from_expr(ns, "", inv));
  return true;
}

/// Classification of the program's recursion structure.
enum class recursiont
{
  none,          // no call-graph cycles — pure loop program
  self_only,     // every cycle is a direct self-loop (f calls f); each
                 // such f is in @ref self_recursive. Tractable: try to
                 // prove each by a ranking function over its parameters.
  unsupported,   // mutual recursion (cycle spanning >1 function) or a
                 // function-pointer call — out of scope, force UNKNOWN.
};

struct recursion_infot
{
  recursiont kind;
  std::unordered_set<irep_idt, irep_id_hash> self_recursive;
};

/// Build the static call graph (skipping body.hide library helpers, which
/// are verification scaffolding handled by the existing machinery — e.g.
/// __ESBMC_atexit_handler's indirect call, linked into every program)
/// and classify the recursion structure. The ranking check reasons about
/// LOOP termination; recursion is an independent non-termination source,
/// so we must either prove each recursive function by its own ranking
/// argument (self-recursion) or bail to UNKNOWN (mutual recursion /
/// function pointers).
recursion_infot analyze_recursion(const goto_functionst &goto_functions)
{
  recursion_infot info;
  info.kind = recursiont::none;

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
      {
        // Function-pointer call: unknown target, can't rule out a cycle.
        info.kind = recursiont::unsupported;
        return info;
      }
    }
  }

  // DFS for cycles. colors: 0=unvisited, 1=on-stack, 2=done. A back-edge
  // to a node on the current stack is a cycle; if that node is the same
  // function (direct self-call) it's self-recursion, otherwise it spans
  // multiple functions (mutual recursion → unsupported).
  std::unordered_map<irep_idt, int, irep_id_hash> color;
  bool mutual = false;
  std::function<void(const irep_idt &)> dfs = [&](const irep_idt &f) {
    color[f] = 1;
    auto it = callees.find(f);
    if (it != callees.end())
    {
      for (const auto &g : it->second)
      {
        if (g == f)
        {
          info.self_recursive.insert(f); // direct self-call
          continue;
        }
        int c = color.count(g) ? color[g] : 0;
        if (c == 1)
          mutual = true; // back-edge across functions
        else if (c == 0 && callees.count(g))
          dfs(g);
      }
    }
    color[f] = 2;
  };

  for (const auto &fn : callees)
    if (color.count(fn.first) == 0)
      dfs(fn.first);

  if (mutual)
    info.kind = recursiont::unsupported;
  else if (!info.self_recursive.empty())
    info.kind = recursiont::self_only;
  return info;
}

/// Prove a directly self-recursive function terminates by a linear
/// ranking function over its parameters. The function must have the
/// shape: a straight-line prefix of forward guard-IFs (each
/// `IF !(g) GOTO L` jumping past the recursive call) leading to one or
/// more recursive call sites; no loops, no other control flow.
///
/// For each recursive call site we form:
///   path_cond  = conjunction of the positive guards g of the IFs fallen
///                through to reach the call;
///   m          = difference measure derived from path_cond's relational
///                atom (the same template as loops);
///   m(args)    = m with each formal parameter replaced by the call-site
///                argument expression (the recursive "transition").
/// and discharge, per call site:
///   bounded:   path_cond ∧ (m < L)         UNSAT
///   decrease:  path_cond ∧ (m(args) >= m)  UNSAT
/// All call sites must pass. Returns false (→ UNKNOWN) on any
/// unsupported shape (loops, memory-touching guards/args, no relational
/// atom, parameter count mismatch).
bool prove_self_recursion_terminates(
  const irep_idt &fname,
  const goto_functiont &fn,
  optionst &options,
  const namespacet &ns)
{
  if (!is_code_type(fn.type))
    return false;
  const code_type2t &ft = to_code_type(fn.type);
  const std::vector<irep_idt> &formals = ft.argument_names;

  // Reject any backwards GOTO: a loop inside the recursive function is a
  // separate non-termination source this per-call ranking doesn't cover.
  for (const auto &ins : fn.body.instructions)
    if (ins.is_backwards_goto())
      return false;

  const auto begin = fn.body.instructions.begin();
  const auto end = fn.body.instructions.end();

  bool found_call = false;
  // Walk to each recursive call site, tracking the path condition as the
  // conjunction of guards of forward-IFs we fall through (i.e. whose
  // jump-over-the-call branch we do NOT take). This is sound only for the
  // straight-line guard-prefix shape; any non-IF branch or unhandled
  // instruction makes us bail.
  std::vector<expr2tc> active_guards; // guards still in scope at this pc
  for (auto it = begin; it != end; ++it)
  {
    if (it->is_function_call())
    {
      const code_function_call2t &call = to_code_function_call2t(it->code);
      if (!is_symbol2t(call.function))
        return false;
      if (to_symbol2t(call.function).thename != fname)
        continue; // call to some other (non-recursive) function: ignore

      // A recursive call site. Build its path condition.
      found_call = true;
      if (call.operands.size() != formals.size())
        return false;

      expr2tc path_cond;
      for (const auto &g : active_guards)
        path_cond = is_nil_expr(path_cond) ? g : and2tc(path_cond, g);
      if (is_nil_expr(path_cond) || touches_memory(path_cond))
        return false;

      // Measure from the path condition's relational atom. For a
      // conjunction we use the first relational conjunct that yields a
      // measure (the guards were collected outermost-first).
      expr2tc m, L;
      bool got = false;
      for (const auto &g : active_guards)
        if (!touches_memory(g) && measure_from_guard(g, m, L))
        {
          got = true;
          break;
        }
      if (!got)
        return false;

      // m(args): substitute each formal parameter symbol by the
      // corresponding call-site argument expression.
      expr2tc m_args = m;
      for (size_t i = 0; i < formals.size(); ++i)
      {
        if (touches_memory(call.operands[i]))
          return false;
        // The formal appears in m as a symbol named like the parameter;
        // build that symbol from the function type's arg name + type.
        expr2tc formal_sym =
          symbol2tc(ft.arguments[i], formals[i]);
        m_args = subst(m_args, formal_sym, call.operands[i]);
      }

      // Obligations for this call site.
      if (!is_unsat(and2tc(path_cond, lessthan2tc(m, L)), options, ns))
        return false;
      if (!is_unsat(
            and2tc(path_cond, greaterthanequal2tc(m_args, m)), options, ns))
        return false;
      continue;
    }

    // Track guard scope on forward IFs: `IF !(cond) GOTO L` contributes
    // the positive guard `!(IF guard)` = cond for the fall-through path
    // up to its target L. We approximate scope as "until end" — sound
    // for the straight-line guard-prefix shape where all IFs jump to the
    // common exit past the call.
    if (it->is_goto())
    {
      if (it->is_backwards_goto() || it->targets.size() != 1)
        return false;
      if (is_true(it->guard))
        continue; // unconditional forward goto: tolerate (rare)
      expr2tc pos = it->guard;
      make_not(pos);
      simplify(pos);
      active_guards.push_back(pos);
      continue;
    }

    // Straight-line bookkeeping/assigns are tolerated but do NOT update
    // the measure (a self-recursion measure over parameters only holds
    // if the parameters aren't reassigned before the call; reject that).
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (is_symbol2t(a.target))
      {
        const irep_idt &tn = to_symbol2t(a.target).thename;
        for (const auto &f : formals)
          if (f == tn)
            return false; // a formal is reassigned before the call
      }
      continue;
    }
    if (
      it->is_skip() || it->is_location() || it->type == DEAD ||
      it->is_end_function() || it->is_return())
      continue;

    return false; // unhandled instruction shape
  }

  return found_call; // proven iff we saw (and discharged) a recursive call
}
} // namespace

tvt try_prove_termination_by_ranking(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns)
{
  // Classify recursion. Mutual recursion / function pointers are out of
  // scope (UNKNOWN). Self-recursion is handled per-function by a ranking
  // argument over the parameters; if any self-recursive function can't
  // be proven, the whole check is inconclusive.
  recursion_infot rec = analyze_recursion(goto_functions);
  if (rec.kind == recursiont::unsupported)
    return tvt(tvt::TV_UNKNOWN);
  for (const irep_idt &fname : rec.self_recursive)
  {
    auto f_it = goto_functions.function_map.find(fname);
    if (
      f_it == goto_functions.function_map.end() ||
      !prove_self_recursion_terminates(
        fname, f_it->second, options, ns))
      return tvt(tvt::TV_UNKNOWN);
  }

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
      if (!prove_loop_terminates(rl, f_it->second.body, options, ns))
        return tvt(tvt::TV_UNKNOWN);
    }
  }

  // All loops proven terminating. With recursion ruled out above and
  // every natural loop ranked, no non-termination source remains
  // (vacuously true if there are no loops).
  return tvt(tvt::TV_TRUE);
}
