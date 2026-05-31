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
  // produces exactly one path (the assignment sequence). A body with N
  // sequential `if (cond)`/`if (cond) else` blocks produces up to 2^N
  // paths (capped). The ranking obligations and invariant inductiveness
  // must hold on EVERY feasible path; failure on any feasible path falls
  // back to UNKNOWN. Each path also carries its path condition (the
  // conjunction of branch selectors that this path requires), so when an
  // infeasible-under-the-supporting-invariant path appears its obligation
  // is vacuously discharged rather than forcing UNKNOWN.
  struct loop_patht
  {
    std::vector<assignt> assigns;
    expr2tc cond; // path condition, defaults to true
  };
  std::vector<loop_patht> paths;
};

// Forward declarations: subst_parallel and apply_body are used by the
// loop recognizer (for dereference substitution and not yet measure
// derivation respectively) but are defined further down alongside the
// other transition-relation helpers. reaches_head is reused by the
// dominance-aware prefix scan below.
expr2tc
subst_parallel(const expr2tc &e, const std::map<expr2tc, expr2tc> &post);
bool reaches_head(
  goto_programt::const_targett start,
  goto_programt::const_targett head,
  goto_programt::const_targett end_it);
void flatten_and(const expr2tc &e, std::vector<expr2tc> &out);
expr2tc apply_body(const expr2tc &e, const std::vector<assignt> &body);

/// Is @p e a relational comparison (>, >=, <, <=) — the guard shape we
/// can derive a difference measure from?
bool is_relational(const expr2tc &e)
{
  return is_greaterthan2t(e) || is_greaterthanequal2t(e) || is_lessthan2t(e) ||
         is_lessthanequal2t(e);
}

/// True iff @p e is a disequality `a != b`. Treated as a "guard shape"
/// for the certifier: while `a != b` holds, the loop runs; either `a -
/// b` or `b - a` must strictly decrease toward 0 — we emit BOTH
/// candidates and prove termination if EITHER discharges.
bool is_disequality(const expr2tc &e)
{
  return is_notequal2t(e);
}

/// True iff @p instr is a `FUNCTION_CALL` to a helper that has no effect
/// on whether the loop terminates. The SV-COMP convention has these
/// helpers either no-op the property entirely (`__VERIFIER_assume` is
/// `assume(c)` — a path filter that doesn't change state or whether the
/// loop runs forever) or end the program via abort if the predicate
/// fails (`__VERIFIER_assert`, `assert`, `reach_error`, `__assert_fail`,
/// `abort`, `exit`). An execution that aborts is *terminating*, so for
/// the termination property these are equivalent to a no-op: they
/// neither preserve nor invent infinite executions, and their
/// counterfactual (where the assert didn't fire) is the path the
/// certifier already analyses.
///
/// We let these survive `collect_straight_line` / Shape B's span
/// collector so loop bodies of the form `foo(); __VERIFIER_assert(P);
/// bar();` are recognized — without this filter the certifier rejects
/// every such loop and the user has to manually strip the assertions
/// before analysis.
bool is_termination_irrelevant_call(const goto_programt::instructiont &instr)
{
  if (!instr.is_function_call())
    return false;
  const code_function_call2t &c = to_code_function_call2t(instr.code);
  if (!is_symbol2t(c.function))
    return false;
  irep_idt fname = to_symbol2t(c.function).thename;
  static const char *allowlist[] = {
    "c:@F@__VERIFIER_assert",
    "c:@F@__VERIFIER_assume",
    "c:@F@assert",
    "c:@F@__assert_fail",
    "c:@F@reach_error",
    "c:@F@abort",
    "c:@F@exit"};
  for (const char *nm : allowlist)
    if (fname == nm)
      return true;
  return false;
}

/// True if @p e contains a memory-dependent subexpression (dereference,
/// array index, struct/union member, byte op). Such expressions can't
/// be handed to the solver directly — they need symex's dereferencing
/// and memory model to be resolved first. We only build obligations
/// from scalar (symbol/constant/arithmetic) expressions, so loops whose
/// guard or transition touch memory are out of scope for this pass and
/// must fall back to UNKNOWN rather than crash the solver.
///
/// The optional @p exempt_derefs set lists `dereference2t` expressions
/// that the caller has proven safe to treat as scalars (e.g. dereferences
/// of loop-invariant pointers with distinct allocation provenance). A
/// deref appearing structurally in @p exempt_derefs is NOT counted as a
/// memory touch — but any other deref, index, or member access still is.
bool touches_memory(
  const expr2tc &e,
  const std::set<expr2tc> &exempt_derefs = {})
{
  if (is_nil_expr(e))
    return false;
  if (is_dereference2t(e) && exempt_derefs.count(e))
    return false;
  if (
    is_dereference2t(e) || is_index2t(e) || is_member2t(e) ||
    is_byte_extract2t(e) || is_byte_update2t(e))
    return true;
  bool found = false;
  e->foreach_operand([&](const expr2tc &op) {
    if (!found)
      found = touches_memory(op, exempt_derefs);
  });
  return found;
}

/// True if @p e contains any `sideeffect2t` subexpression (e.g. a NONDET
/// call result, a malloc, a function-call result). Such expressions are
/// opaque to our solver layer — bv-encoding a free sideeffect either
/// stalls or behaves nondeterministically depending on the backend — so
/// callers that need to feed expressions to the solver must reject them.
/// Used by the path-condition builder after apply_body substitution may
/// have inlined a `temp = NONDET(...)` prefix assignment into the IF
/// guard.
bool contains_sideeffect(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_sideeffect2t(e))
    return true;
  bool found = false;
  e->foreach_operand([&](const expr2tc &op) {
    if (!found)
      found = contains_sideeffect(op);
  });
  return found;
}

/// Collect every `dereference2t` subexpression appearing in @p e into
/// @p out (keyed structurally). Other memory operations (index, member,
/// byte ops) are not collected — the caller separately rejects loops
/// that contain them. We only look at dereferences because the only
/// memory-handling extension supported here is treating a loop-invariant
/// `*p` as a scalar.
void collect_derefs(const expr2tc &e, std::set<expr2tc> &out)
{
  if (is_nil_expr(e))
    return;
  if (is_dereference2t(e))
  {
    out.insert(e);
    // Don't recurse into the pointer expression — we only care that the
    // deref ITSELF can be treated as a scalar; the pointer being indexed
    // is not part of the scalar arithmetic.
    return;
  }
  e->foreach_operand([&](const expr2tc &op) { collect_derefs(op, out); });
}

/// Peel surrounding `typecast2t`s off @p e, returning the innermost
/// non-typecast expression. Useful for inspecting an rhs like
/// `(int*) malloc(...)` where the cast is incidental to the analysis.
const expr2tc &peel_typecasts(const expr2tc &e)
{
  const expr2tc *cur = &e;
  while (!is_nil_expr(*cur) && is_typecast2t(*cur))
    cur = &to_typecast2t(*cur).from;
  return *cur;
}

/// Walk the function prefix from entry to the loop @p head along the
/// single dominator path and build a map from each symbol's name to the
/// rhs of its (typecast-peeled) assignment on that path. The walk is the
/// same single-path traversal collect_seeds uses: at each conditional IF
/// we ask reaches_head whether the fall-through or the jump branch leads
/// to the loop head; if exactly one does, we follow it. If both branches
/// reach the head the prefix isn't a single dominator path (a merge of
/// two assignments to the same pointer could give a non-dominating fact)
/// and we bail. The has_invalid flag is set on any unsupported prefix
/// instruction (FUNCTION_CALL, OTHER, RETURN, back-edge, multi-target
/// GOTO, jump target reached from outside an accepted IF), at which point
/// callers (notably the deref-substitution check) treat the prefix as
/// unsafe and refuse the substitution.
struct prefix_defst
{
  std::map<irep_idt, expr2tc> defs;
  bool has_invalid = false;
};
prefix_defst
scan_prefix_defs(const goto_programt &body, goto_programt::const_targett head)
{
  prefix_defst out;
  auto end_it = body.instructions.end();
  std::set<unsigned> justified_targets;

  for (auto it = body.instructions.begin(); it != head;)
  {
    if (it->is_target() && !justified_targets.count(it->location_number))
    {
      // Reached a merge point we didn't justify by an accepted IF jump:
      // some other edge can land here, so the dominator-path assumption
      // breaks.
      out.has_invalid = true;
      return out;
    }
    if (
      it->is_skip() || it->is_location() || it->type == DECL ||
      it->type == DEAD)
    {
      ++it;
      continue;
    }
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (is_symbol2t(a.target))
        out.defs[to_symbol2t(a.target).thename] = peel_typecasts(a.source);
      ++it;
      continue;
    }
    if (
      it->is_goto() && !is_true(it->guard) && !it->is_backwards_goto() &&
      it->targets.size() == 1)
    {
      auto tgt = it->targets.front();
      bool fall_reaches = reaches_head(std::next(it), head, end_it);
      bool jump_reaches = reaches_head(tgt, head, end_it);
      if (!fall_reaches && !jump_reaches)
      {
        out.has_invalid = true;
        return out;
      }
      if (fall_reaches && !jump_reaches)
      {
        ++it; // follow fall-through
        continue;
      }
      if (jump_reaches && !fall_reaches)
      {
        justified_targets.insert(tgt->location_number);
        it = tgt; // follow the jump
        continue;
      }
      // Both branches reach the head: a merge before the loop where two
      // assignments could disagree on a tracked pointer's value. Bail.
      out.has_invalid = true;
      return out;
    }
    // FUNCTION_CALL, OTHER, RETURN, back-edge, multi-target GOTO, etc.:
    // any of these may mutate memory or break the single-path walk in a
    // way we don't track.
    out.has_invalid = true;
    return out;
  }
  return out;
}

/// Trace pointer @p p back through the prefix's symbol chain to its
/// "memory-cell identity" — the address-of'd local's name, or the
/// allocation sideeffect expression's address (uniquely identified by the
/// expr2tc pointer value). Returns a string identifier if successful, or
/// empty string if the chain doesn't reduce to a fresh allocation.
/// Follows up to a small fixed depth so it stays linear.
std::string pointer_cell_identity(const irep_idt &p, const prefix_defst &defs)
{
  if (defs.has_invalid)
    return {};
  irep_idt cur = p;
  for (int depth = 0; depth < 8; ++depth)
  {
    auto it = defs.defs.find(cur);
    if (it == defs.defs.end())
      return {};
    const expr2tc &src = it->second; // typecast-peeled
    if (is_sideeffect2t(src))
    {
      auto k = to_sideeffect2t(src).kind;
      if (
        k != sideeffect2t::allockind::malloc &&
        k != sideeffect2t::allockind::realloc &&
        k != sideeffect2t::allockind::alloca)
        return {};
      // The sideeffect's expression pointer is a unique id for the
      // allocation site (no two prefix instructions share the same
      // expr2tc object).
      char buf[32];
      std::snprintf(
        buf, sizeof(buf), "alloc:%p", static_cast<const void *>(src.get()));
      return buf;
    }
    if (is_address_of2t(src))
    {
      const expr2tc &target = to_address_of2t(src).ptr_obj;
      // Identity is the address-of'd object's printable form. For a plain
      // local symbol that's its name; richer lvalues (a[i].f) are not
      // pointer-disjoint in general — refuse them.
      if (!is_symbol2t(target))
        return {};
      return "local:" + id2string(to_symbol2t(target).thename);
    }
    if (is_symbol2t(src))
    {
      cur = to_symbol2t(src).thename;
      continue;
    }
    return {};
  }
  return {};
}

/// Collect straight-line assigns in [@p first, @p last). Returns false if
/// any instruction is anything other than DECL/ASSIGN/skip/location/DEAD,
/// or a memory-touching assign — the caller treats that as an unrecognized
/// body shape. ASSIGNs are appended to @p out in program order. The
/// @p exempt_derefs set lists dereferences already proven safe to treat as
/// scalars (see compute_safe_derefs); they don't count as memory touches.
bool collect_straight_line(
  goto_programt::const_targett first,
  goto_programt::const_targett last,
  std::vector<assignt> &out,
  const std::set<expr2tc> &exempt_derefs = {})
{
  for (auto it = first; it != last; ++it)
  {
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (
        touches_memory(a.target, exempt_derefs) ||
        touches_memory(a.source, exempt_derefs))
        return false;
      out.push_back({a.target, a.source});
      continue;
    }
    if (
      it->is_skip() || it->is_location() || it->type == DEAD ||
      it->type == DECL)
      continue;
    // FUNCTION_CALL to a termination-irrelevant helper (assert, assume,
    // abort, exit, ...) — skip; see `is_termination_irrelevant_call`.
    if (is_termination_irrelevant_call(*it))
      continue;
    return false;
  }
  return true;
}

/// Inspect every expression mentioned in the loop body (instructions
/// (head, back)) plus the loop's guard; collect all dereference2t
/// subexpressions; check that EVERY dereference is of the shape `*p`
/// where p is a plain symbol with distinct-allocation provenance from
/// the function prefix, and that p is not assigned anywhere in the loop
/// body. On success, populate @p out_exempt with the set of safe
/// dereferences and @p out_map with a substitution from each deref to a
/// fresh free symbol of the appropriate scalar type — the rest of the
/// pipeline runs on the scalar-substituted expressions and never sees a
/// dereference. Returns true on success; on failure (any deref not safe)
/// returns false with both outputs empty, and the caller proceeds with
/// the original (memory-touching) rejection.
///
/// Soundness rationale: each loop-invariant pointer dereferences a fixed
/// memory cell every iteration, so morally `*p` IS a scalar. Distinct
/// allocations (malloc/realloc/alloca/address_of of distinct locals)
/// cannot alias each other by C semantics, so substituting each `*p` with
/// a DISTINCT fresh symbol is sound even when several derefs appear.
bool compute_safe_derefs(
  goto_programt::const_targett head,
  goto_programt::const_targett back,
  const expr2tc &guard,
  const goto_programt &fn_body,
  std::set<expr2tc> &out_exempt,
  std::map<expr2tc, expr2tc> &out_map)
{
  out_exempt.clear();
  out_map.clear();

  // Walk every instruction in (head, back) and collect derefs from the
  // expressions we'll later analyse (ASSIGN target/source, IF guard).
  std::set<expr2tc> derefs;
  collect_derefs(guard, derefs);
  for (auto it = std::next(head); it != back; ++it)
  {
    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      collect_derefs(a.target, derefs);
      collect_derefs(a.source, derefs);
    }
    else if (it->is_goto() && !is_nil_expr(it->guard))
      collect_derefs(it->guard, derefs);
  }
  if (derefs.empty())
    return true; // nothing to do; existing pipeline handles it

  // Identify every body-assigned scalar symbol — pointers in this set
  // are NOT loop-invariant, so their derefs are unsafe.
  std::set<irep_idt> body_assigned;
  for (auto it = std::next(head); it != back; ++it)
  {
    if (!it->is_assign())
      continue;
    const code_assign2t &a = to_code_assign2t(it->code);
    if (is_symbol2t(a.target))
      body_assigned.insert(to_symbol2t(a.target).thename);
  }

  // Validate each deref: pointer must be a plain symbol, loop-invariant,
  // have distinct-allocation provenance in the prefix, AND its memory cell
  // must be disjoint from every other dereffed pointer's cell. Two
  // pointers reaching the same allocation (or both `&` of the same local)
  // would alias at runtime, so substituting them by distinct fresh scalars
  // would be unsound.
  prefix_defst defs = scan_prefix_defs(fn_body, head);
  std::set<std::string> seen_cells;
  for (const expr2tc &d : derefs)
  {
    const dereference2t &deref = to_dereference2t(d);
    if (!is_symbol2t(deref.value))
      return false;
    const irep_idt &p = to_symbol2t(deref.value).thename;
    if (body_assigned.count(p))
      return false;
    std::string cell = pointer_cell_identity(p, defs);
    if (cell.empty())
      return false;
    if (!seen_cells.insert(cell).second)
      return false; // two pointers share this cell — aliasing risk
  }

  // Build the substitution: each unique deref maps to a fresh free symbol
  // of the dereferenced type. The name is salted with the deref node's
  // raw pointer value so collisions with program-defined symbols (or with
  // synthetic symbols from other recognize_loop invocations) are
  // essentially impossible — the prefix `$rank_deref$` is not a legal C
  // identifier and the hex tail makes each fresh symbol unique.
  for (const expr2tc &d : derefs)
  {
    const dereference2t &deref = to_dereference2t(d);
    const irep_idt &p = to_symbol2t(deref.value).thename;
    char buf[64];
    std::snprintf(
      buf, sizeof(buf), "$rank_deref$%p$", static_cast<const void *>(d.get()));
    std::string sym_name = std::string(buf) + id2string(p);
    out_map[d] = symbol2tc(d->type, sym_name);
  }
  out_exempt = std::move(derefs);
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
bool recognize_loop(
  const loopst &loop,
  const goto_programt &fn_body,
  ranking_loopt &out)
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

  // Compute safe dereferences: any *p where p is a loop-invariant pointer
  // with distinct-allocation provenance from the function prefix. Each is
  // substituted by a fresh scalar symbol, turning a pointer-as-scalar loop
  // (e.g. `while (*p >= 0) (*p)--;` after `p = malloc(...)`) into the
  // ordinary scalar shape the rest of the pipeline already handles. If the
  // analysis fails for any deref, we treat the loop as memory-touching and
  // fall back to UNKNOWN (the substitution either succeeds for all derefs
  // or not at all — partial substitution would be unsound).
  std::set<expr2tc> exempt_derefs;
  std::map<expr2tc, expr2tc> deref_map;
  if (!compute_safe_derefs(
        out.head, out.back, pos_guard, fn_body, exempt_derefs, deref_map))
    return false;

  // The guard is either a relational atom or a top-level conjunction of
  // relational atoms. We require at least one usable relational atom (so
  // measure_candidates_from_guard yields >= 1 candidate). Disjunctive
  // guards (`||` at the top level) are not handled and will fail here.
  if (touches_memory(pos_guard, exempt_derefs))
    return false;
  if (
    !is_relational(pos_guard) && !is_disequality(pos_guard) &&
    !is_and2t(pos_guard))
    return false;
  out.guard =
    deref_map.empty() ? pos_guard : subst_parallel(pos_guard, deref_map);

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
  auto rewrite = [&](std::vector<assignt> &path) {
    if (deref_map.empty())
      return;
    for (assignt &a : path)
    {
      a.lhs = subst_parallel(a.lhs, deref_map);
      a.rhs = subst_parallel(a.rhs, deref_map);
    }
  };

  if (!has_internal_goto)
  {
    std::vector<assignt> path;
    if (!collect_straight_line(body_begin, out.back, path, exempt_derefs))
      return false;
    rewrite(path);
    ranking_loopt::loop_patht p;
    p.assigns = std::move(path);
    p.cond = gen_true_expr();
    out.paths.push_back(std::move(p));
    return true;
  }

  // Shape (B): a sequence of straight-line spans and if/else blocks at
  // top level. Parse the body into a list of "blocks", each either one
  // straight-line span or one if/else pair, then enumerate paths as the
  // cartesian product. Refuse if the projected path count would exceed a
  // small cap (the solver pays per-path obligations, and the underlying
  // invariant inductiveness check pays per-path-per-atom).
  struct block_t
  {
    bool is_span; // span vs. if/else
    std::vector<assignt> span;
    std::vector<assignt> then_arm, else_arm; // if !is_span
    expr2tc if_cond; // raw IF guard (NEGATION of the C-source if-condition):
      // then-arm runs when !if_cond holds, else-arm when if_cond holds.
  };
  std::vector<block_t> blocks;
  static constexpr size_t kMaxPaths = 16;

  // const iterator so we can assign jump targets (which are const_targett)
  // back into it when we skip past an if/else block.
  goto_programt::const_targett it = body_begin;
  goto_programt::const_targett back_it = out.back;
  while (it != back_it)
  {
    // Skip skip/location/DECL/DEAD that aren't significant.
    if (
      it->is_skip() || it->is_location() || it->type == DEAD ||
      it->type == DECL)
    {
      ++it;
      continue;
    }
    if (it->is_assign() || is_termination_irrelevant_call(*it))
    {
      // Greedily collect a straight-line span up to the next GOTO or end.
      // Termination-irrelevant FUNCTION_CALLs (assert/assume/abort/...)
      // count as part of the span — collect_straight_line will skip
      // over them and they vanish from the path's assigns.
      auto span_begin = it;
      while (it != out.back && !it->is_goto() &&
             (it->is_assign() || it->is_skip() || it->is_location() ||
              it->type == DEAD || it->type == DECL ||
              is_termination_irrelevant_call(*it)))
        ++it;
      block_t b;
      b.is_span = true;
      if (!collect_straight_line(span_begin, it, b.span, exempt_derefs))
        return false;
      blocks.push_back(std::move(b));
      continue;
    }
    if (
      it->is_goto() && !is_true(it->guard) && !it->is_backwards_goto() &&
      it->targets.size() == 1)
    {
      // An if/else block. Same layout as before: IF !cond GOTO else_label;
      // <then>; [GOTO merge; else_label: <else>;] merge_label: ...
      auto if_target = it->targets.front();
      if (if_target->location_number <= it->location_number)
        return false; // not forward
      if (touches_memory(it->guard, exempt_derefs))
        return false;
      auto then_begin = std::next(it);
      auto then_end = then_begin;
      while (then_end != if_target && !then_end->is_goto())
        ++then_end;
      goto_programt::const_targett merge_label;
      if (then_end == if_target)
        merge_label = if_target; // no else
      else
      {
        if (
          !then_end->is_goto() || !is_true(then_end->guard) ||
          then_end->is_backwards_goto() || then_end->targets.size() != 1)
          return false;
        merge_label = then_end->targets.front();
        if (merge_label->location_number <= then_end->location_number)
          return false;
      }
      // The merge must not lie PAST the back-edge (that would be outside
      // the loop body). The merge IS the back-edge in the no-post-assigns
      // case (e.g. `while (G) { if (X) S1 else S2 }` with nothing after
      // the if/else); that's accepted — the outer while loop simply has
      // no post-arm.
      if (merge_label->location_number > out.back->location_number)
        return false;
      block_t b;
      b.is_span = false;
      b.if_cond = it->guard;
      if (!collect_straight_line(
            then_begin, then_end, b.then_arm, exempt_derefs))
        return false;
      if (
        then_end != if_target &&
        !collect_straight_line(
          if_target, merge_label, b.else_arm, exempt_derefs))
        return false;
      blocks.push_back(std::move(b));
      it = merge_label;
      continue;
    }
    // Any other instruction shape (backwards/multi-target GOTO, RETURN,
    // FUNCTION_CALL, OTHER, etc.) — not handled.
    return false;
  }

  // Bound the projected number of paths up front. Each if/else doubles the
  // count; spans don't. Refusing high path counts protects the solver from
  // a body with many branches that would explode obligation work.
  size_t n_paths = 1;
  for (const block_t &b : blocks)
    if (!b.is_span)
    {
      if (n_paths > kMaxPaths / 2)
        return false;
      n_paths *= 2;
    }

  // Enumerate paths by binary mask over the if/else blocks. Mask bit i is
  // 0 for "take the i-th if/else's THEN arm", 1 for "take the ELSE arm".
  // The path is the concatenation of every span and the chosen arm of
  // every if/else, in program order.
  size_t n_ifs = 0;
  std::vector<size_t> if_indices; // indices into blocks of the if/else blocks
  for (size_t i = 0; i < blocks.size(); ++i)
    if (!blocks[i].is_span)
    {
      if_indices.push_back(i);
      ++n_ifs;
    }
  for (size_t mask = 0; mask < (size_t(1) << n_ifs); ++mask)
  {
    std::vector<assignt> path;
    std::vector<expr2tc> cond_atoms;
    for (size_t i = 0; i < blocks.size(); ++i)
    {
      const block_t &b = blocks[i];
      if (b.is_span)
      {
        path.insert(path.end(), b.span.begin(), b.span.end());
        continue;
      }
      // Find which bit in mask corresponds to this if/else.
      size_t which =
        std::find(if_indices.begin(), if_indices.end(), i) - if_indices.begin();
      bool take_else = (mask >> which) & 1;
      const std::vector<assignt> &arm = take_else ? b.else_arm : b.then_arm;
      // Path-condition atom for this IF — evaluated at the point the IF
      // executes, which is AFTER the assignments accumulated in `path` so
      // far. Substitute those assignments into the raw IF guard via
      // apply_body, turning a post-prior-block-state expression into a
      // pre-iteration-state expression that conjoins correctly with the
      // rest of the path's pre-state path condition. Without this
      // substitution, an earlier `y = y - 1` followed by `if (y > 0)`
      // would record `y > 0` (pre-state) when the IF actually sees
      // `(y-1) > 0` (i.e. pre-state `y > 1`) — a feasible runtime path
      // might then be reported as infeasible and a non-decreasing
      // obligation discharged vacuously (potential wrong-true).
      if (!is_nil_expr(b.if_cond))
      {
        expr2tc atom = apply_body(b.if_cond, path);
        if (!take_else)
          make_not(atom);
        simplify(atom);
        if (deref_map.size())
          atom = subst_parallel(atom, deref_map);
        // Skip the atom if substitution inlined an opaque sideeffect
        // (e.g. a temporary `temp = NONDET(...)` assignment in a prior
        // block was inlined into the IF guard). The solver can stall on
        // such expressions; conservatively treating the path-condition
        // as `true` is sound (it just makes the obligation harder to
        // discharge, never easier — never vacuously discharging).
        if (!contains_sideeffect(atom))
          cond_atoms.push_back(atom);
      }
      path.insert(path.end(), arm.begin(), arm.end());
    }
    rewrite(path);
    ranking_loopt::loop_patht p;
    p.assigns = std::move(path);
    if (cond_atoms.empty())
      p.cond = gen_true_expr();
    else
    {
      expr2tc c = cond_atoms.front();
      for (size_t k = 1; k < cond_atoms.size(); ++k)
        c = and2tc(c, cond_atoms[k]);
      p.cond = c;
    }
    out.paths.push_back(std::move(p));
  }
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
expr2tc subst_parallel(const expr2tc &e, const std::map<expr2tc, expr2tc> &post)
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
  auto rec2 =
    [&](const expr2tc &a, const expr2tc &b) -> std::pair<expr2tc, expr2tc> {
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
/// Build the widened-int64 difference `(int64)a - (int64)b` if both
/// operands are scalar BV that we can safely lift without wraparound.
/// Returns true on success and writes the lifted expression to @p out;
/// returns false (and leaves @p out untouched) if either operand is
/// not a BV or a 64-bit non-constant operand could produce a
/// difference outside int64. Used both by `measure_from_relational`
/// (which produces a single candidate from a `<`/`<=`/`>`/`>=` atom)
/// and by `measure_from_disequality` (which produces two candidates
/// from a `!=` atom — one for each potential direction).
bool make_widened_difference(
  const expr2tc &a_in,
  const expr2tc &b_in,
  expr2tc &out)
{
  expr2tc a = peel_typecasts(a_in);
  expr2tc b = peel_typecasts(b_in);

  if (!is_bv_type(a->type) || !is_bv_type(b->type))
    return false;

  // 32-bit source operands give a difference in [-(2^32-1), 2^32-1],
  // far inside int64. A 64-bit source operand could overflow int64 —
  // refuse unless the wide side is a constant fitting in int32.
  auto wide_side_is_int32_constant = [](const expr2tc &e) -> bool {
    if (e->type->get_width() <= 32)
      return true;
    if (!is_constant_int2t(e))
      return false;
    const BigInt &v = to_constant_int2t(e).value;
    BigInt lo(-2147483648LL), hi(2147483647LL);
    return v >= lo && v <= hi;
  };
  if (!wide_side_is_int32_constant(a) || !wide_side_is_int32_constant(b))
    return false;

  type2tc wide = get_int_type(64);
  out = sub2tc(wide, typecast2tc(wide, a), typecast2tc(wide, b));
  return true;
}

/// Build a single candidate (m, L) from one relational atom @p atom.
/// Returns false if the atom is not relational or its operands are not
/// scalar BV of width <= 32. The measure is m = (int64)a - (int64)b with
/// L = 0 or 1 depending on strict/non-strict.
bool measure_from_relational(const expr2tc &atom, expr2tc &m, expr2tc &L)
{
  expr2tc a, b;
  BigInt bound;
  if (is_greaterthan2t(atom))
  {
    a = to_greaterthan2t(atom).side_1;
    b = to_greaterthan2t(atom).side_2;
    bound = 1;
  }
  else if (is_greaterthanequal2t(atom))
  {
    a = to_greaterthanequal2t(atom).side_1;
    b = to_greaterthanequal2t(atom).side_2;
    bound = 0;
  }
  else if (is_lessthan2t(atom))
  {
    a = to_lessthan2t(atom).side_2;
    b = to_lessthan2t(atom).side_1;
    bound = 1;
  }
  else if (is_lessthanequal2t(atom))
  {
    a = to_lessthanequal2t(atom).side_2;
    b = to_lessthanequal2t(atom).side_1;
    bound = 0;
  }
  else
    return false;

  if (!make_widened_difference(a, b, m))
    return false;
  type2tc wide = get_int_type(64);
  L = constant_int2tc(wide, bound);
  return true;
}

/// Build TWO candidates (m, L) from a disequality atom `a != b`. The
/// guard `a != b` holds iff `a > b` OR `a < b`; we don't know which,
/// so we emit both `(a-b, 1)` and `(b-a, 1)`. The candidate-discharge
/// loop tries each independently; whichever direction admits a strict
/// decrease + bounded-below proof yields termination.
///
/// Returns false if the operands are not scalar BV of width <= 32, or
/// the lifted differences would overflow int64 (see
/// `make_widened_difference`).
bool measure_from_disequality(
  const expr2tc &atom,
  std::vector<std::pair<expr2tc, expr2tc>> &out)
{
  if (!is_notequal2t(atom))
    return false;
  const notequal2t &neq = to_notequal2t(atom);
  expr2tc m_ab, m_ba;
  if (!make_widened_difference(neq.side_1, neq.side_2, m_ab))
    return false;
  if (!make_widened_difference(neq.side_2, neq.side_1, m_ba))
    return false;
  type2tc wide = get_int_type(64);
  expr2tc L = constant_int2tc(wide, BigInt(1));
  out.push_back({m_ab, L});
  out.push_back({m_ba, L});
  return true;
}

/// Build ranking candidates from a loop guard. A single relational atom
/// yields one candidate. A top-level `&&` conjunction yields one
/// candidate per relational conjunct (any conjunct whose decrease and
/// boundedness obligations BOTH discharge proves the loop terminates --
/// the guard `g1 && g2` is false as soon as either conjunct becomes
/// false, so making either rank-derived measure decrease to its floor is
/// enough). Returns false if the guard yields no usable candidate.
///
/// Disjunctive (||) guards are out of scope by design (full-correctness
/// would need per-disjunct path-split obligations rivalling the cstr*
/// pointer-stride work in complexity); they fall through here.
bool measure_candidates_from_guard(
  const expr2tc &guard,
  std::vector<std::pair<expr2tc, expr2tc>> &out)
{
  std::vector<expr2tc> atoms;
  flatten_and(guard, atoms);
  for (const expr2tc &atom : atoms)
  {
    expr2tc m, L;
    if (measure_from_relational(atom, m, L))
    {
      out.push_back({m, L});
      continue;
    }
    // Disequality `a != b` yields two direction candidates.
    measure_from_disequality(atom, out);
  }
  return !out.empty();
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
prefix_seedst
collect_seeds(const goto_programt &body, goto_programt::const_targett head)
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
    if (
      it->is_goto() && !is_true(it->guard) && !it->is_backwards_goto() &&
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
    expr2tc atom;                // pre-state atom, holds at entry
    std::vector<expr2tc> primes; // post-state image per body path
  };
  // Build one post-substitution map per body path: each `post[i][v]` is the
  // value of `v` after path `i`, expressed in pre-state terms. The atom
  // inductiveness check then needs the atom preserved by EVERY path.
  std::vector<std::map<expr2tc, expr2tc>> path_posts(rl.paths.size());
  for (size_t i = 0; i < rl.paths.size(); ++i)
    for (const auto &a : rl.paths[i].assigns)
      path_posts[i][a.lhs] = subst_parallel(a.rhs, path_posts[i]);

  auto post_state_of = [&](const expr2tc &e, size_t i) {
    return widen_arith(subst_parallel(e, path_posts[i]));
  };

  // Atoms are built for every scalar variable that could matter to the
  // proof: the union of body-modified lhs symbols AND any symbol that
  // appears in a path condition. The latter is essential because a
  // bound on a never-written variable can still rule out an infeasible
  // path's obligation (e.g. `debug = 0` in the prefix lets the path
  // through `if (debug != 0)` be discharged vacuously).
  std::map<irep_idt, expr2tc> body_vars;
  auto register_symbol = [&](const expr2tc &e) {
    if (is_symbol2t(e) && is_bv_type(e->type) && e->type->get_width() <= 32)
      body_vars.emplace(to_symbol2t(e).thename, e);
  };
  std::function<void(const expr2tc &)> collect_path_symbols =
    [&](const expr2tc &e) {
      if (is_nil_expr(e))
        return;
      register_symbol(e);
      e->foreach_operand(collect_path_symbols);
    };
  for (const auto &p : rl.paths)
  {
    for (const auto &a : p.assigns)
      register_symbol(a.lhs);
    collect_path_symbols(p.cond);
  }

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
    auto make_atom = [&](const expr2tc &x, const expr2tc &y) -> expr2tc {
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
    // If the source atom is strict (`>` or `<`), also try its non-strict
    // counterpart (`>=` / `<=`). The strict one implies the non-strict
    // one, so the non-strict version holds at loop entry; and the non-
    // strict version is typically what's inductive across a decrementing
    // body (`while (x != 0) x = x - 1;` with seed `x > 0` keeps `x >= 0`
    // — `x > 0` itself fails inductiveness when x reaches 1). The
    // fixpoint loop drops whichever variant doesn't survive.
    if (is_greaterthan2t(a) || is_lessthan2t(a))
    {
      auto make_relaxed = [&](const expr2tc &x, const expr2tc &y) -> expr2tc {
        if (is_greaterthan2t(a))
          return greaterthanequal2tc(x, y);
        return lessthanequal2tc(x, y);
      };
      cand rc;
      rc.atom = make_relaxed(l, r);
      for (const auto &lr : lr_per_path)
        rc.primes.push_back(make_relaxed(lr.first, lr.second));
      atoms.push_back(std::move(rc));
    }
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
      for (size_t pi = 0; pi < atoms[i].primes.size(); ++pi)
      {
        // Path infeasible under the invariant ⇒ preservation vacuous.
        expr2tc neg = atoms[i].primes[pi];
        make_not(neg);
        expr2tc check = and2tc(and2tc(inv, rl.guard), rl.paths[pi].cond);
        check = and2tc(check, neg);
        if (!is_unsat(check, options, ns))
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
  // Candidate measures. A simple relational guard yields one candidate; a
  // top-level conjunction yields one per relational conjunct. Termination
  // is proved if ANY candidate satisfies both the bounded-below and the
  // strict-decrease obligations under the synthesized invariant (since
  // the loop guard `g1 && g2 && ...` becomes false as soon as ANY
  // conjunct does, having one rank reach its floor suffices).
  std::vector<std::pair<expr2tc, expr2tc>> candidates;
  if (!measure_candidates_from_guard(rl.guard, candidates))
    return false;

  // Supporting invariant: an inductive conjunction of constant bounds that
  // holds on entry. Sound to assume in the obligations below (it is true
  // every iteration). For loops that need no extra facts this is just true.
  expr2tc inv = synthesize_invariant(rl, body, options, ns);

  for (const auto &cand : candidates)
  {
    const expr2tc &m = cand.first;
    const expr2tc &L = cand.second;

    // Bounded-below obligation: inv ∧ guard ∧ (m < L) must be UNSAT.
    if (!is_unsat(
          and2tc(and2tc(inv, rl.guard), lessthan2tc(m, L)), options, ns))
      continue;

    // Decrease obligation: m strictly decreases on every feasible body
    // path. A path whose condition is infeasible under the invariant has
    // the obligation discharged vacuously.
    bool decreases = true;
    for (const auto &p : rl.paths)
    {
      expr2tc m_prime = apply_body(m, p.assigns);
      expr2tc obligation = and2tc(and2tc(inv, rl.guard), p.cond);
      obligation = and2tc(obligation, greaterthanequal2tc(m_prime, m));
      if (!is_unsat(obligation, options, ns))
      {
        decreases = false;
        break;
      }
    }
    if (!decreases)
      continue;

    log_debug(
      "termination",
      "ranking function proved loop terminates: measure={} invariant={}",
      from_expr(ns, "", m),
      from_expr(ns, "", inv));
    return true;
  }

  // 2-D lexicographic fallback. For each ordered pair (m1, m2) of
  // candidate measures (m1 != m2), prove that on every body path
  //   m1' < m1   OR   (m1' == m1 AND m2' < m2)
  // and that the lexicographic floor (m1 < L1) OR (m1 == L1 AND m2 < L2)
  // makes the guard unsatisfiable. This catches two-counter loops
  // common in termination-crafted (c.03: `while (x < y && z < INT_MAX)
  // { if (x < z) x++; else z++; }` — neither x nor z decreases on every
  // path, but (y - x, INT_MAX - z) decreases lex on both).
  //
  // We iterate ordered pairs (i != j) so both directions are tried:
  // either m1 = guard_atom_i / m2 = guard_atom_j, or the swap. With
  // few candidates (typically 2–3 per guard conjunction) the search
  // is cheap — at most ~6 pair attempts per loop, each costing one
  // bounded-below SAT and `|paths|` decrease checks. A pathological
  // guard with many && conjuncts (or many disequalities each
  // doubled into two direction candidates) could push the pair
  // count quadratically, so cap the candidate pool the lex pass
  // considers at a small constant — beyond that we bail to UNKNOWN
  // rather than burn solver time exploring an exponential of pairs
  // the user is unlikely to have written by hand.
  static constexpr size_t kMaxLexCandidates = 8;
  if (candidates.size() >= 2 && candidates.size() <= kMaxLexCandidates)
  {
    for (size_t i = 0; i < candidates.size(); ++i)
    {
      for (size_t j = 0; j < candidates.size(); ++j)
      {
        if (i == j)
          continue;
        const expr2tc &m1 = candidates[i].first;
        const expr2tc &L1 = candidates[i].second;
        const expr2tc &m2 = candidates[j].first;
        const expr2tc &L2 = candidates[j].second;

        // Lex floor: guard must be unsatisfiable when m1 has reached
        // L1 and (if m1 == L1) m2 has reached L2.
        expr2tc floor = or2tc(
          lessthan2tc(m1, L1),
          and2tc(equality2tc(m1, L1), lessthan2tc(m2, L2)));
        if (!is_unsat(and2tc(and2tc(inv, rl.guard), floor), options, ns))
          continue;

        bool decreases = true;
        for (const auto &p : rl.paths)
        {
          expr2tc m1p = apply_body(m1, p.assigns);
          expr2tc m2p = apply_body(m2, p.assigns);
          // Lex-decrease: m1' < m1, or m1' == m1 and m2' < m2.
          expr2tc decr = or2tc(
            lessthan2tc(m1p, m1),
            and2tc(equality2tc(m1p, m1), lessthan2tc(m2p, m2)));
          expr2tc obligation = and2tc(and2tc(inv, rl.guard), p.cond);
          obligation = and2tc(obligation, not2tc(decr));
          if (!is_unsat(obligation, options, ns))
          {
            decreases = false;
            break;
          }
        }
        if (!decreases)
          continue;

        log_debug(
          "termination",
          "ranking proved (lex 2D): m1={} m2={} invariant={}",
          from_expr(ns, "", m1),
          from_expr(ns, "", m2),
          from_expr(ns, "", inv));
        return true;
      }
    }
  }

  return false;
}

/// Classification of the program's recursion structure.
enum class recursiont
{
  none,        // no call-graph cycles — pure loop program
  self_only,   // every cycle is a direct self-loop (f calls f); each
               // such f is in @ref self_recursive. Tractable: try to
               // prove each by a ranking function over its parameters.
  unsupported, // mutual recursion (cycle spanning >1 function) or a
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
        if (!touches_memory(g) && measure_from_relational(g, m, L))
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
        expr2tc formal_sym = symbol2tc(ft.arguments[i], formals[i]);
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
      !prove_self_recursion_terminates(fname, f_it->second, options, ns))
      return tvt(tvt::TV_UNKNOWN);
  }

  // Every loop in every function with a body must be proven terminating
  // for the program to be declared terminating. A single loop we cannot
  // handle makes the whole check inconclusive.
  //
  // Bare unconditional self-loops (`A: goto A;`) are intentionally NOT
  // added to goto_loopst's loop list when --termination is on (see
  // goto_loops.cpp ~line 60, gated against the assume(false) rewrite
  // that would otherwise erase them). They are nevertheless real
  // non-terminating constructs, so we walk every function's
  // instructions directly and refuse to certify termination if any are
  // present — the verdict belongs to the non-termination pass /
  // k-induction's forward condition, not to us.
  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available || f_it->second.body.hide)
      continue;
    for (const auto &instr : f_it->second.body.instructions)
    {
      if (!instr.is_backwards_goto() || instr.targets.size() != 1)
        continue;
      auto tgt = instr.targets.front();
      // Bare self-loop: `1: GOTO 1` (target == this instruction itself).
      // Use location_number for identity since goto_programt iterators
      // are unstable across some passes; identical location_number on a
      // backwards goto's target is the definition the frontend uses.
      if (tgt->location_number == instr.location_number)
        return tvt(tvt::TV_UNKNOWN);
    }
  }

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
      if (!recognize_loop(loop, f_it->second.body, rl))
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
