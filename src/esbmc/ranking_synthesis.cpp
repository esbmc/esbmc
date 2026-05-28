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
#include <map>
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
/// invariant bounds. Comparisons/boolean structure are preserved; only the
/// integer arithmetic is widened. Non-integer or memory expressions are
/// returned unchanged (callers only pass scalar integer expressions).
expr2tc widen_arith(const expr2tc &e)
{
  if (is_nil_expr(e))
    return e;
  type2tc wide = get_int_type(64);
  if (is_symbol2t(e) || is_constant_int2t(e))
    return typecast2tc(wide, e);
  // Arithmetic nodes: rebuild in the wide type with widened operands, so
  // the operation is performed in 64 bits (no 32-bit wraparound).
  if (is_add2t(e))
    return add2tc(
      wide, widen_arith(to_add2t(e).side_1), widen_arith(to_add2t(e).side_2));
  if (is_sub2t(e))
    return sub2tc(
      wide, widen_arith(to_sub2t(e).side_1), widen_arith(to_sub2t(e).side_2));
  if (is_mul2t(e))
    return mul2tc(
      wide, widen_arith(to_mul2t(e).side_1), widen_arith(to_mul2t(e).side_2));
  if (is_neg2t(e))
    return neg2tc(wide, widen_arith(to_neg2t(e).value));
  // Anything else (e.g. an existing cast): value-preserve into wide.
  return typecast2tc(wide, e);
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

/// Collect a constant seed for each body-modified scalar variable from the
/// loop's pre-header. The base case of an assumed invariant requires the
/// seed to hold on EVERY first entry to the loop head, so we only trust a
/// constant assignment that provably reaches the head with no intervening
/// re-definition, branch merge, call, or aliasing write. We enforce this
/// conservatively: the prefix from function entry up to the loop head must
/// be a single straight-line block — only DECL/ASSIGN/skip/location, no
/// GOTO/IF, no FUNCTION_CALL, no jump target landing inside it, and no
/// assignment whose target or source touches memory (a pointer write could
/// alias a tracked variable). If any of those appear we return an empty
/// seed map (the loop gets no supporting invariant and falls back to the
/// bare ranking check). Within a clean prefix, seed[v] is the value of the
/// LAST `ASSIGN v = <int constant>` before the head.
std::map<irep_idt, BigInt> collect_seeds(
  const goto_programt &body,
  goto_programt::const_targett head)
{
  std::map<irep_idt, BigInt> seed;
  for (auto it = body.instructions.begin(); it != head; ++it)
  {
    // A jump target inside the prefix means another edge reaches here, so
    // a syntactically-preceding constant need not hold on that path.
    if (it->is_target())
      return {};
    if (it->is_skip() || it->is_location() || it->type == DECL)
      continue;
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (touches_memory(a.target) || touches_memory(a.source))
        return {}; // a pointer write here could alias a tracked variable
      if (!is_symbol2t(a.target))
        return {};
      const irep_idt &name = to_symbol2t(a.target).thename;
      if (is_constant_int2t(a.source))
        seed[name] = to_constant_int2t(a.source).value; // last write wins
      else
        seed.erase(name); // non-constant redefinition: no trusted seed
      continue;
    }
    // GOTO / IF / FUNCTION_CALL / anything else: the prefix is not a clean
    // straight-line dominator block, so we cannot trust syntactic seeds.
    return {};
  }
  return seed;
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
  std::map<irep_idt, BigInt> seed = collect_seeds(body, rl.head);
  if (seed.empty())
    return gen_true_expr();

  // Build candidate atoms over the int64 domain so the body's arithmetic on
  // these bounds cannot wrap: a 32-bit increment like d1 = d2 + 1 wraps
  // under modular bitvector semantics at INT_MAX, which would spuriously
  // break an otherwise-inductive bound. Operands are at most 32-bit
  // (checked), so widened values stay well inside int64. For each
  // body-modified scalar variable with a constant seed c, both v >= c and
  // v <= c hold at entry. We track each atom alongside its post-state image
  // (the bound on v rewritten with v's widened post-iteration value) so the
  // inductiveness check reasons entirely in the non-wrapping wide domain.
  struct cand
  {
    expr2tc atom;  // bound on the (widened) variable, holds at entry
    expr2tc prime; // same bound on the variable's widened post-state value
  };
  std::vector<cand> atoms;
  for (const auto &a : rl.body)
  {
    if (!is_symbol2t(a.lhs) || !is_bv_type(a.lhs->type))
      continue;
    if (a.lhs->type->get_width() > 32)
      continue;
    auto s = seed.find(to_symbol2t(a.lhs).thename);
    if (s == seed.end())
      continue;
    type2tc wide = get_int_type(64);
    expr2tc c = constant_int2tc(wide, s->second);
    expr2tc v = widen_arith(a.lhs);                       // (int64)v
    expr2tc vp = widen_arith(apply_body(a.lhs, rl.body)); // (int64)post(v)
    atoms.push_back({greaterthanequal2tc(v, c), greaterthanequal2tc(vp, c)});
    atoms.push_back({lessthanequal2tc(v, c), lessthanequal2tc(vp, c)});
  }
  if (atoms.empty())
    return gen_true_expr();

  // Fixpoint: drop atoms the body does not preserve under the guard and the
  // other surviving atoms. An atom A survives iff
  //   (∧ atoms) ∧ guard ∧ ¬A'   is UNSAT,
  // where A' is the atom's post-state image. Atoms are only ever removed,
  // so this terminates.
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
      expr2tc neg = atoms[i].prime;
      make_not(neg);
      if (!is_unsat(and2tc(and2tc(inv, rl.guard), neg), options, ns))
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

  // m' = m after one loop body iteration.
  expr2tc m_prime = apply_body(m, rl.body);

  // Supporting invariant: an inductive conjunction of constant bounds that
  // holds on entry. Sound to assume in the obligations below (it is true
  // every iteration). For loops that need no extra facts this is just true.
  expr2tc inv = synthesize_invariant(rl, body, options, ns);

  // Bounded-below obligation: inv ∧ guard ∧ (m < L) must be UNSAT, i.e.
  // under the invariant the guard implies m >= L (so the measure cannot
  // decrease past the floor without the guard becoming false → loop exits).
  if (!is_unsat(and2tc(and2tc(inv, rl.guard), lessthan2tc(m, L)), options, ns))
    return false;

  // Decrease obligation: inv ∧ guard ∧ ¬(m' < m) must be UNSAT, i.e. under
  // the invariant and guard the measure strictly decreases every iteration.
  if (!is_unsat(
        and2tc(and2tc(inv, rl.guard), greaterthanequal2tc(m_prime, m)),
        options,
        ns))
    return false;

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
