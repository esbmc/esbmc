#include <unordered_set>
#include <cassert>
#include <functional>
#include <goto-symex/engine/goto_symex.h>
#include <goto-symex/state/goto_symex_state.h>
#include <goto-symex/symex_invariant.h>
#include <goto-symex/equation/symex_target_equation.h>
#include <langapi/language_util.h>
#include <solvers/smt/smt_conv.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/message/message.h>
#include <util/arith/mp_arith.h>
#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>
#include <util/irep/migrate.h>
#include <util/irep/std_expr.h>

namespace
{
struct equation_conversion_statet
{
  // Conjunction of all in-scope assumptions, as an expr2tc. Solver-AST
  // handling stays inside smt_convt: the equation only manipulates
  // expr2tc and lets convert_ast/assert_expr bridge to the solver.
  expr2tc assumpt_expr;
  // Negated discharge condition per kept assertion (or the path
  // assumption alone in vacuity mode). OR'd together and asserted once.
  std::vector<expr2tc> assertions;
};

void pre_register_addresses(
  smt_convt &smt_conv,
  symex_target_equationt::SSA_stepst::iterator begin,
  symex_target_equationt::SSA_stepst::iterator end)
{
  // Only pre-register address_of of compile-time constants (string and
  // array literals).  These have static lifetime and exist throughout the
  // program, so including them early in the address space cannot produce
  // spurious candidate matches for int-to-ptr casts -- any int-to-ptr
  // cast could legitimately reach them regardless of where the literal's
  // use happens to appear in the source.  Dynamic/automatic objects keep
  // their original lazy registration to avoid exposing later-allocated
  // memory to earlier casts.
  // Shared subexpressions make an SSA step a DAG, not a tree -- a propagated
  // `with` chain over a nested array references itself once per store -- so an
  // unmemoised walk costs a number of paths exponential in the store count.
  std::unordered_set<const expr2t *> seen;
  std::function<void(const expr2tc &)> walk = [&](const expr2tc &e) {
    if (!e)
      return;
    if (!seen.insert(e.get()).second)
      return;
    if (is_address_of2t(e))
    {
      // Unwrap index/member chains (e.g. &""[0]) to reach the literal base.
      expr2tc obj = to_address_of2t(e).ptr_obj;
      while (is_index2t(obj) || is_member2t(obj))
        obj = is_index2t(obj) ? to_index2t(obj).source_value
                              : to_member2t(obj).source_value;
      if (is_constant_string2t(obj) || is_constant_array2t(obj))
        smt_conv.convert_ast(e);
    }
    e->foreach_operand([&](const expr2tc &op) { walk(op); });
  };

  for (auto it = begin; it != end; ++it)
  {
    const symex_target_equationt::SSA_stept &step = *it;
    if (step.ignore)
      continue;
    walk(step.guard);
    walk(step.cond);
    walk(step.lhs);
    walk(step.rhs);
    if (step.output_data)
      for (const expr2tc &arg : step.output_data->output_args)
        walk(arg);
  }
}

/// PathReachable asks only whether the path is feasible, ignoring the claim:
/// UNSAT over the kept claims means every discharge was vacuous. Satisfiable
/// asks whether the claim can hold, in the same shape the violation query
/// uses, so that negating the implication forces its antecedents -- conjoining
/// `cond` instead would be satisfied by leaving the path untaken, which makes
/// every claim look satisfiable (issue #7585).
void encode_assertion(
  equation_conversion_statet &state,
  symex_target_equationt::SSA_stept &step,
  symex_target_equationt::assertion_modet mode)
{
  using mode_t = symex_target_equationt::assertion_modet;

  if (mode == mode_t::PathReachable)
  {
    step.cond_expr = state.assumpt_expr;
    state.assertions.push_back(state.assumpt_expr);
    return;
  }

  const expr2tc &claim =
    mode == mode_t::Satisfiable ? step.cond_neg : step.cond;
  step.cond_expr = implies2tc(state.assumpt_expr, claim);
  state.assertions.push_back(not2tc(step.cond_expr));
}

void convert_internal_step(
  const namespacet &ns,
  bool ssa_trace,
  bool ssa_smt_trace,
  unsigned &output_count,
  smt_convt &smt_conv,
  equation_conversion_statet &state,
  symex_target_equationt::SSA_stept &step,
  symex_target_equationt::assertion_modet mode)
{
  if (step.ignore)
  {
    step.cond_expr = gen_true_expr();
    return;
  }

  if (ssa_trace)
  {
    std::ostringstream oss;
    step.output(ns, oss);
    log_status("{}", oss.str());
  }

  smt_conv.begin_step(step.guard, step.cond);

  if (step.is_assume() || step.is_assert() || step.is_branching())
  {
    smt_conv.convert_ast(step.cond);
    if (ssa_smt_trace)
      smt_conv.dump_expr(step.cond);
  }
  else if (step.is_assignment())
  {
    smt_conv.note_assignment(step.lhs, step.rhs);
    smt_conv.convert_assign(step.cond);
    if (ssa_smt_trace)
      smt_conv.dump_expr(step.cond);
  }
  else if (step.is_output())
  {
    symex_target_equationt::SSA_stept::output_datat &od = step.output_payload();
    for (const expr2tc &tmp : od.output_args)
    {
      if (is_constant_expr(tmp) || is_constant_string2t(tmp))
        od.converted_output_args.push_back(tmp);
      else
      {
        expr2tc sym =
          symbol2tc(tmp->type, "symex::output::" + i2string(output_count++));
        expr2tc eq = equality2tc(sym, tmp);
        smt_conv.convert_assign(eq);
        if (ssa_smt_trace)
          smt_conv.dump_expr(eq);
        od.converted_output_args.push_back(sym);
      }
    }
  }
  else if (step.is_renumber())
  {
    smt_conv.renumber_symbol_address(step.guard, step.lhs, step.rhs);
  }
  else if (!step.is_skip())
  {
    assert(0 && "Unexpected SSA step type in conversion");
  }
  smt_conv.end_step();

  if (step.is_assert())
  {
    encode_assertion(state, step, mode);
  }
  else if (step.is_assume())
  {
    state.assumpt_expr = and2tc(state.assumpt_expr, step.cond);
  }
  else
  {
    step.cond_expr = gen_true_expr();
  }
}
} // namespace

// =============================================================================
// Surface-C byte-WITH rewrite: when a heap dyn-object's WITH chain has
// byte-decomposed a typed pointer write, recover the typed pointer source from
// CONCAT-of-byte-extracts read sites by walking the chain (through if-merges)
// and rewrite the SSA step's cond to reference the typed pointer directly.
// This restores pointer provenance for downstream pointer_object / SAME-OBJECT.
//
// Design: notes/napp/heap_byte_provenance/fix_directions.md (§5).
// Stage-0 verdict: notes/napp/heap_byte_provenance/surface_c_stage0_probe.md.
//
// Sound by construction: every gate failure leaves the original byte-CONCAT in
// place (today's KNOWNBUG behavior); the rewrite only fires when the chain
// resolves to a coherent typed-pointer or if-tree-of-typed-pointers source.
// =============================================================================
namespace
{
namespace ByteWithRewrite
{
using DynObjDefs = std::unordered_map<std::string, const expr2tc *>;

static expr2tc strip_casts(const expr2tc &e)
{
  expr2tc cur = e;
  while (!is_nil_expr(cur) && (is_bitcast2t(cur) || is_typecast2t(cur)))
  {
    if (is_bitcast2t(cur))
      cur = to_bitcast2t(cur).from;
    else
      cur = to_typecast2t(cur).from;
  }
  return cur;
}

static bool is_dyn_obj_symbol(const expr2tc &e)
{
  if (!is_symbol2t(e))
    return false;
  const std::string &name = to_symbol2t(e).get_symbol_name();
  return name.find("symex_dynamic::dynamic_") != std::string::npos ||
         name.find("symex_dynamic::realloc_") != std::string::npos;
}

static std::string get_full_key(const expr2tc &sym)
{
  if (!is_symbol2t(sym))
    return "";
  return to_symbol2t(sym).get_symbol_name();
}

// Match: outer pointer-typed cast wrapping a concat. The concat is what we
// rewrite; the outer cast preserves type-compatibility with the surrounding
// expression. We only fire when the concat is consumed in a pointer context.
static bool match_outer_ptr_cast_concat(
  const expr2tc &e,
  expr2tc &concat_out,
  type2tc &target_ptr_type_out)
{
  expr2tc inner;
  type2tc target;
  if (is_typecast2t(e))
  {
    if (!is_pointer_type(e->type))
      return false;
    inner = to_typecast2t(e).from;
    target = e->type;
  }
  else if (is_bitcast2t(e))
  {
    if (!is_pointer_type(e->type))
      return false;
    inner = to_bitcast2t(e).from;
    target = e->type;
  }
  else
  {
    return false;
  }
  if (!is_concat2t(inner))
    return false;
  concat_out = inner;
  target_ptr_type_out = target;
  return true;
}

// Match concat2t(... bitcast(uchar, index2t(symbol, K)) ...) where all leaves
// reference the same dyn-obj symbol. Ordered offsets are returned in CONCAT
// traversal order (MSB-first for big-endian encoding of the concat).
static bool match_byte_concat_on_dynobj(
  const expr2tc &e,
  expr2tc &dyn_obj_symbol_out,
  std::vector<BigInt> &offsets_out)
{
  if (!is_concat2t(e))
    return false;
  std::vector<expr2tc> leaves;
  std::function<bool(const expr2tc &)> collect = [&](const expr2tc &n) -> bool {
    if (is_concat2t(n))
    {
      const concat2t &c = to_concat2t(n);
      return collect(c.side_1) && collect(c.side_2);
    }
    leaves.push_back(n);
    return true;
  };
  if (!collect(e))
    return false;
  if (leaves.size() < 2)
    return false;

  expr2tc first_sym;
  for (const expr2tc &leaf : leaves)
  {
    expr2tc inner = strip_casts(leaf);
    if (!is_index2t(inner))
      return false;
    const index2t &idx = to_index2t(inner);
    if (!is_constant_int2t(idx.index))
      return false;
    if (!is_dyn_obj_symbol(idx.source_value))
      return false;
    if (is_nil_expr(first_sym))
      first_sym = idx.source_value;
    else if (get_full_key(first_sym) != get_full_key(idx.source_value))
      return false;
    offsets_out.push_back(to_constant_int2t(idx.index).value);
  }
  dyn_obj_symbol_out = first_sym;
  return true;
}

enum class WalkResult
{
  RESOLVED_LIT,
  RESOLVED_IF,
  UNRESOLVED,
  IF_BLOWUP,
  CHAIN_BLOWUP
};

// Per-byte resolution: the typed pointer source value AND the byte_extract
// source_offset (so we can verify offset progression).
struct ByteResolution
{
  expr2tc typed_source; // For RESOLVED_LIT: the typed pointer. For RESOLVED_IF:
                        // an if2tc tree.
  BigInt source_offset; // Only meaningful for RESOLVED_LIT leaves; if-tree
                        // per-arm offsets must agree.
  WalkResult result;
};

// Walk one byte chain backward from `current` to find the typed pointer source
// written at `target_offset` of the dyn-obj.
//
// Returns ByteResolution with:
//   - For RESOLVED_LIT: typed_source = byte_extract source value (pointer-typed
//     after the soundness gate); source_offset = byte_extract source_offset.
//   - For RESOLVED_IF: typed_source = if2tc(cond, P_t, P_f); source_offset
//     unused at top level (per-arm offsets are inside the if2t).
//   - Otherwise: typed_source = nil.
static ByteResolution walk_one_byte_chain(
  const expr2tc &current,
  const DynObjDefs &defs,
  const BigInt &target_offset,
  int depth,
  int if_depth)
{
  ByteResolution out;
  out.result = WalkResult::UNRESOLVED;

  if (depth > 256)
  {
    out.result = WalkResult::CHAIN_BLOWUP;
    return out;
  }
  if (if_depth > 16)
  {
    out.result = WalkResult::IF_BLOWUP;
    return out;
  }

  if (is_symbol2t(current))
  {
    auto it = defs.find(get_full_key(current));
    if (it == defs.end())
      return out; // UNRESOLVED
    return walk_one_byte_chain(
      *it->second, defs, target_offset, depth + 1, if_depth);
  }

  if (is_with2t(current))
  {
    const with2t &w = to_with2t(current);
    if (
      is_constant_int2t(w.update_field) &&
      to_constant_int2t(w.update_field).value == target_offset)
    {
      expr2tc inner = strip_casts(w.update_value);
      if (is_byte_extract2t(inner))
      {
        const byte_extract2t &be = to_byte_extract2t(inner);
        // Soundness gate: source must be pointer-typed after strip_casts.
        // Rejects risk5 (int-to-ptr).
        expr2tc be_src = strip_casts(be.source_value);
        if (!is_pointer_type(be_src->type))
          return out; // UNRESOLVED
        if (!is_constant_int2t(be.source_offset))
          return out; // symbolic offset; bail
        out.typed_source = be_src;
        out.source_offset = to_constant_int2t(be.source_offset).value;
        out.result = WalkResult::RESOLVED_LIT;
        return out;
      }
      return out; // UNRESOLVED — non-byte-extract update value
    }
    return walk_one_byte_chain(
      w.source_value, defs, target_offset, depth + 1, if_depth);
  }

  if (is_if2t(current))
  {
    const if2t &iff = to_if2t(current);
    ByteResolution r_t = walk_one_byte_chain(
      iff.true_value, defs, target_offset, depth + 1, if_depth + 1);
    ByteResolution r_f = walk_one_byte_chain(
      iff.false_value, defs, target_offset, depth + 1, if_depth + 1);

    if (is_nil_expr(r_t.typed_source) || is_nil_expr(r_f.typed_source))
    {
      if (
        r_t.result == WalkResult::IF_BLOWUP ||
        r_f.result == WalkResult::IF_BLOWUP)
        out.result = WalkResult::IF_BLOWUP;
      else if (
        r_t.result == WalkResult::CHAIN_BLOWUP ||
        r_f.result == WalkResult::CHAIN_BLOWUP)
        out.result = WalkResult::CHAIN_BLOWUP;
      return out;
    }

    // Both arms resolved. The per-arm source_offsets must match (same byte of
    // both pointer sources is being extracted).
    if (
      r_t.result == WalkResult::RESOLVED_LIT &&
      r_f.result == WalkResult::RESOLVED_LIT &&
      r_t.source_offset != r_f.source_offset)
      return out; // UNRESOLVED — per-arm offsets disagree

    if (r_t.typed_source == r_f.typed_source)
    {
      // Both arms identical — collapse the if.
      out.typed_source = r_t.typed_source;
      out.source_offset = r_t.source_offset;
      out.result = (r_t.result == WalkResult::RESOLVED_IF ||
                    r_f.result == WalkResult::RESOLVED_IF)
                     ? WalkResult::RESOLVED_IF
                     : WalkResult::RESOLVED_LIT;
      return out;
    }

    // Build if-tree of typed sources. Both arms must have compatible types.
    if (r_t.typed_source->type != r_f.typed_source->type)
      return out; // UNRESOLVED — type mismatch
    out.typed_source = if2tc(
      r_t.typed_source->type, iff.cond, r_t.typed_source, r_f.typed_source);
    out.source_offset = r_t.source_offset; // implicitly = r_f.source_offset
    out.result = WalkResult::RESOLVED_IF;
    return out;
  }

  return out; // UNRESOLVED — unknown chain link
}

// Verify the byte_extract source_offsets line up with the CONCAT byte order.
// For the canonical 8-byte little-endian pointer encoding, CONCAT leaves are
// listed MSB-first: dyn-obj offsets [k+7, k+6, ..., k+0]. The byte_extracts
// pull bytes [7, 6, ..., 0] from the typed pointer's representation. So leaf i
// (0-based, MSB=0) has dyn-obj offset (k + width-1 - i) and byte_extract
// source_offset (width-1 - i). The two are off by a constant k. Check that
// (dyn_obj_offset - byte_extract_source_offset) is the same across all leaves.
static bool check_source_offset_progression(
  const std::vector<BigInt> &concat_offsets,
  const std::vector<ByteResolution> &per_byte)
{
  if (concat_offsets.size() != per_byte.size())
    return false;
  if (concat_offsets.empty())
    return false;

  BigInt expected_delta = concat_offsets[0] - per_byte[0].source_offset;
  for (size_t i = 1; i < concat_offsets.size(); ++i)
  {
    BigInt delta = concat_offsets[i] - per_byte[i].source_offset;
    if (delta != expected_delta)
      return false;
  }
  return true;
}

// Verify that all per-byte resolutions resolve to the SAME typed source (modulo
// strip_casts). For RESOLVED_IF results, the typed_source is an if2t whose
// arms must agree across bytes.
static bool
check_typed_source_coherence(const std::vector<ByteResolution> &per_byte)
{
  if (per_byte.empty())
    return false;
  expr2tc canonical = strip_casts(per_byte[0].typed_source);
  for (size_t i = 1; i < per_byte.size(); ++i)
  {
    if (strip_casts(per_byte[i].typed_source) != canonical)
      return false;
  }
  return true;
}

// High-level: try to recover the typed pointer source for a matched
// byte-CONCAT. Returns nil expr on failure (any soundness gate). On success,
// returns the typed pointer (or if2tc-of-typed-pointers) — pointer-typed.
static expr2tc try_recover(
  const expr2tc &concat,
  const expr2tc &dyn_obj_symbol,
  const std::vector<BigInt> &concat_offsets,
  const DynObjDefs &defs)
{
  std::vector<ByteResolution> per_byte;
  per_byte.reserve(concat_offsets.size());

  for (const BigInt &offset : concat_offsets)
  {
    ByteResolution r = walk_one_byte_chain(dyn_obj_symbol, defs, offset, 0, 0);
    if (
      r.result != WalkResult::RESOLVED_LIT &&
      r.result != WalkResult::RESOLVED_IF)
      return expr2tc();
    if (is_nil_expr(r.typed_source))
      return expr2tc();
    per_byte.push_back(r);
  }

  if (!check_source_offset_progression(concat_offsets, per_byte))
    return expr2tc();
  if (!check_typed_source_coherence(per_byte))
    return expr2tc();

  // All gates passed. Take the first resolution as the canonical typed source
  // (coherence guarantees all are equivalent).
  (void)concat; // unused; kept for future verbose logging
  return per_byte[0].typed_source;
}

// Recursively walk an expression tree; when we find a pointer-typed cast
// wrapping a CONCAT-of-byte-extracts on a dyn-obj, replace the wrapper with a
// typecast of the recovered typed pointer.
static expr2tc rewrite_recursively(
  const expr2tc &e,
  const DynObjDefs &defs,
  bool &any_change,
  unsigned &fired)
{
  if (is_nil_expr(e))
    return e;

  // Outer pointer-cast wrapping byte-CONCAT-of-byte-extracts.
  expr2tc inner_concat;
  type2tc target_ptr_type;
  if (match_outer_ptr_cast_concat(e, inner_concat, target_ptr_type))
  {
    expr2tc dyn_sym;
    std::vector<BigInt> concat_offsets;
    if (match_byte_concat_on_dynobj(inner_concat, dyn_sym, concat_offsets))
    {
      expr2tc recovered =
        try_recover(inner_concat, dyn_sym, concat_offsets, defs);
      if (!is_nil_expr(recovered))
      {
        any_change = true;
        ++fired;
        log_debug(
          "byte-with-rewrite",
          "rewrote concat-of-byte-extracts on '{}' to typed pointer source",
          get_full_key(dyn_sym));
        if (recovered->type == target_ptr_type)
          return recovered;
        return typecast2tc(target_ptr_type, recovered);
      }
      else
      {
        log_debug(
          "byte-with-rewrite",
          "try_recover FAILED for site '{}'",
          get_full_key(dyn_sym));
      }
    }
  }

  // Recurse into children with bottom-up clone.
  std::vector<expr2tc> new_children;
  bool child_changed = false;
  e->foreach_operand([&](const expr2tc &child) {
    bool local_change = false;
    expr2tc rewritten = rewrite_recursively(child, defs, local_change, fired);
    if (local_change)
      child_changed = true;
    new_children.push_back(rewritten);
  });
  if (!child_changed)
    return e;

  any_change = true;
  expr2tc cloned = e->clone();
  size_t i = 0;
  cloned->Foreach_operand(
    [&](expr2tc &child_ref) { child_ref = new_children[i++]; });
  return cloned;
}

static bool rewrite_byte_concat_on_step(
  symex_target_equationt::SSA_stept &step,
  const DynObjDefs &defs,
  unsigned &fired)
{
  if (is_nil_expr(step.cond))
    return false;
  bool changed = false;
  expr2tc new_cond = rewrite_recursively(step.cond, defs, changed, fired);
  if (!changed)
    return false;
  step.cond = new_cond;
  return true;
}

// Not currently wired into the pipeline; kept compiled to preserve the
// experimental rewrite (see notes/napp/heap_byte_provenance/) until it
// is either promoted to a real pass or excised.
[[maybe_unused]] void run_pass(symex_target_equationt &eq)
{
  DynObjDefs defs;
  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assignment())
      continue;
    if (is_nil_expr(step.lhs) || !is_symbol2t(step.lhs))
      continue;
    if (is_dyn_obj_symbol(step.lhs))
    {
      defs[get_full_key(step.lhs)] = &step.rhs;
    }
  }

  unsigned fired = 0;
  unsigned steps_changed = 0;
  for (auto &step : eq.SSA_steps)
  {
    if (step.ignore)
      continue;
    if (rewrite_byte_concat_on_step(step, defs, fired))
      ++steps_changed;
  }

  if (fired > 0 || steps_changed > 0)
    log_debug(
      "byte-with-rewrite",
      "Surface-C: rewrote {} concat-of-byte-extracts site(s) across {} step(s) "
      "(dyn_obj_defs: {} entries)",
      fired,
      steps_changed,
      defs.size());
}

} // namespace ByteWithRewrite
} // anonymous namespace
// =============================================================================
// END Surface-C byte-WITH rewrite
// =============================================================================

void symex_target_equationt::debug_print_step(const SSA_stept &step) const
{
  std::ostringstream oss;
  step.output(ns, oss);
  log_debug("ssa", "{}", oss.str());
}

void symex_target_equationt::assignment(
  const expr2tc &guard,
  const expr2tc &lhs,
  const expr2tc &original_lhs,
  const expr2tc &rhs,
  const expr2tc &original_rhs,
  const sourcet &source,
  std::vector<stack_framet> stack_trace,
  const bool hidden,
  unsigned loop_number)
{
  assert(!is_nil_expr(lhs));

  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.lhs = lhs;
  SSA_step.original_lhs = original_lhs;
  SSA_step.original_rhs = original_rhs;
  SSA_step.rhs = rhs;
  SSA_step.hidden = hidden;
  SSA_step.cond = equality2tc(lhs, rhs);
  SSA_step.type = symex_targett::ASSIGNMENT;
  SSA_step.source = source;
  if (!stack_trace.empty())
    SSA_step.stack_trace_payload() = std::move(stack_trace);
  SSA_step.loop_number = loop_number;

  if (debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::output(
  const expr2tc &guard,
  const sourcet &source,
  const std::string &fmt,
  const std::list<expr2tc> &args)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.type = symex_targett::OUTPUT;
  SSA_step.source = source;
  auto &od = SSA_step.output_payload();
  od.output_args = args;
  od.format_string = fmt;

  if (debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::branching(
  const expr2tc &guard,
  const expr2tc &cond,
  const sourcet &source,
  const bool hidden,
  unsigned loop_number)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.hidden = hidden;
  SSA_step.type = symex_targett::BREANCHING;
  SSA_step.source = source;
  SSA_step.loop_number = loop_number;

  if (debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::assumption(
  const expr2tc &guard,
  const expr2tc &cond,
  const sourcet &source,
  unsigned loop_number)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type = symex_targett::ASSUME;
  SSA_step.source = source;
  SSA_step.loop_number = loop_number;

  if (debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::assertion(
  const expr2tc &guard,
  const expr2tc &cond,
  const expr2tc &cond_neg,
  const std::string &msg,
  std::vector<stack_framet> stack_trace,
  const sourcet &source,
  unsigned loop_number)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.cond_neg = cond_neg;
  SSA_step.type = symex_targett::ASSERT;
  SSA_step.source = source;
  SSA_step.comment = msg;
  if (!stack_trace.empty())
    SSA_step.stack_trace_payload() = std::move(stack_trace);
  SSA_step.loop_number = loop_number;

  if (debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::renumber(
  const expr2tc &guard,
  const expr2tc &symbol,
  const expr2tc &size,
  const sourcet &source)
{
  assert(is_symbol2t(symbol));
  assert(is_bv_type(size));
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.lhs = symbol;
  SSA_step.rhs = size;
  SSA_step.type = symex_targett::RENUMBER;
  SSA_step.source = source;

  if (debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::convert(smt_convt &smt_conv, assertion_modet mode)
{
  // Pre-register address-of'd string/array literals so int-to-ptr casts see
  // them regardless of source-level declaration order (see
  // pre_register_addresses for scope).
  pre_register_addresses(smt_conv, SSA_steps.begin(), SSA_steps.end());

  // Show the solver every division's operand pair up front: the
  // modulus conversion lowers a remainder compositionally only when
  // the same operands are divided somewhere in the formula.
  for (const auto &SSA_step : SSA_steps)
  {
    smt_conv.note_division_operands(SSA_step.guard);
    smt_conv.note_division_operands(SSA_step.rhs);
    smt_conv.note_division_operands(SSA_step.cond);
  }

  equation_conversion_statet state;
  state.assumpt_expr = gen_true_expr();

  for (auto &SSA_step : SSA_steps)
    convert_internal_step(
      ns,
      ssa_trace,
      ssa_smt_trace,
      output_count,
      smt_conv,
      state,
      SSA_step,
      mode);

  if (!state.assertions.empty())
    smt_conv.assert_expr(disjunction(state.assertions));
}

void symex_target_equationt::output(std::ostream &out) const
{
  for (const auto &SSA_step : SSA_steps)
  {
    SSA_step.output(ns, out);
    out << "--------------"
        << "\n";
  }
}

void symex_target_equationt::short_output(std::ostream &out, bool show_ignored)
  const
{
  for (const auto &SSA_step : SSA_steps)
  {
    SSA_step.short_output(ns, out, show_ignored);
  }
}

void symex_target_equationt::SSA_stept::dump() const
{
  std::ostringstream oss;
  output(*migrate_namespace_lookup, oss);
  log_status("{}", oss.str());
}

void symex_target_equationt::SSA_stept::output(
  const namespacet &ns,
  std::ostream &out) const
{
  if (source.is_set)
  {
    out << "Thread " << source.thread_nr;

    if (source.pc->location.is_not_nil())
      out << " " << source.pc->location << "\n";
    else
      out << "\n";
  }

  switch (type)
  {
  case symex_targett::ASSERT:
    out << "ASSERT"
        << "\n";
    break;
  case symex_targett::ASSUME:
    out << "ASSUME"
        << "\n";
    break;
  case symex_targett::OUTPUT:
    out << "OUTPUT"
        << "\n";
    break;
  case symex_targett::BREANCHING:
    out << "BRANCHING"
        << "\n";
    break;
  case symex_targett::ASSIGNMENT:
    out << "ASSIGNMENT (";
    out << (hidden ? "HIDDEN" : "") << ")\n";
    break;

  default:
    assert(
      type == symex_targett::SKIP && config.options.get_bool_option("ltl"));
  }

  if (is_assert() || is_assume() || is_assignment() || is_branching())
    out << from_expr(ns, "", migrate_expr_back(cond)) << "\n";

  if (is_assert())
    out << comment << "\n";

  if (config.options.get_bool_option("ssa-guards"))
    out << "Guard: " << from_expr(ns, "", migrate_expr_back(guard)) << "\n";
}

void symex_target_equationt::SSA_stept::short_output(
  const namespacet &ns,
  std::ostream &out,
  bool show_ignored) const
{
  if ((is_assignment() || is_assert() || is_assume()) && show_ignored == ignore)
  {
    out << from_expr(ns, "", cond) << "\n";
  }
  else if (is_renumber())
  {
    out << "renumber: " << from_expr(ns, "", lhs) << "\n";
  }
}

void symex_target_equationt::push_ctx()
{
}

void symex_target_equationt::pop_ctx()
{
}

std::ostream &
operator<<(std::ostream &out, const symex_target_equationt &equation)
{
  equation.output(out);
  return out;
}

bool symex_target_equationt::check_for_duplicate_assigns() const
{
  std::map<std::string, unsigned int> countmap;

  for (const auto &SSA_step : SSA_steps)
    if (SSA_step.is_assignment())
    {
      // `assignment()` documents lhs as a symbol but only checks it is
      // non-nil. Skipping a non-symbol lhs would silently narrow the check
      // this function exists to perform, so state the precondition instead.
      SYMEX_INVARIANT(
        is_symbol2t(SSA_step.lhs), "an assignment step's lhs is not a symbol");
      countmap[to_symbol2t(SSA_step.lhs).get_symbol_name()]++;
    }

  bool well_formed = true;
  for (const auto &[name, count] : countmap)
    if (count != 1)
    {
      log_error("SSA name \"{}\" is defined {} times", name, count);
      well_formed = false;
    }

  return well_formed;
}

unsigned int symex_target_equationt::clear_assertions()
{
  unsigned int num_asserts = 0;

  for (SSA_stepst::iterator it = SSA_steps.begin(); it != SSA_steps.end(); ++it)
  {
    if (it->type == symex_targett::ASSERT)
    {
      SSA_stepst::iterator it2 = it;
      --it;
      SSA_steps.erase(it2);
      num_asserts++;
    }
  }

  return num_asserts;
}

// To be used by reconstruct_symbolic_expression
void symex_target_equationt::replace_rec(
  const SSA_stept &step,
  expr2tc &e,
  bool keep_local) const
{
  assert(step.is_assignment());
  if (is_symbol2t(e))
  {
    const std::string lhs_name = to_symbol2t(step.lhs).get_symbol_name();
    if (keep_local && lhs_name.find("goto_symex::") == std::string::npos)
      return;

    if (lhs_name == to_symbol2t(e).get_symbol_name())
      e = step.rhs;
  }

  e->Foreach_operand([&step, &keep_local, this](expr2tc &inner) {
    replace_rec(step, inner, keep_local);
  });
}

void symex_target_equationt::reconstruct_symbolic_expression(
  expr2tc &expr,
  bool keep_local_variables) const
{
  for (auto rit = SSA_steps.rbegin(); rit != SSA_steps.rend(); rit++)
  {
    if (!rit->is_assignment())
      continue;

    replace_rec(*rit, expr, keep_local_variables);
  }
}

struct runtime_encoded_equationt::solver_statet
{
  std::list<equation_conversion_statet> states;
};

runtime_encoded_equationt::runtime_encoded_equationt(
  const namespacet &_ns,
  smt_convt &_conv)
  : symex_target_equationt(_ns),
    conv(_conv),
    solver_state(std::make_unique<solver_statet>())
{
  solver_state->states.emplace_back();
  solver_state->states.back().assumpt_expr = gen_true_expr();
  cvt_progress = SSA_steps.end();
}

runtime_encoded_equationt::~runtime_encoded_equationt() = default;

void runtime_encoded_equationt::flush_latest_instructions()
{
  if (SSA_steps.size() == 0)
    return;

  SSA_stepst::iterator run_it = cvt_progress;
  // Scenarios:
  // * We're at the start of running, in which case cvt_progress == end
  // * We're in the middle, but nothing is left to push, so run_it + 1 == end
  // * We're in the middle, and there's more to convert.
  if (run_it == SSA_steps.end())
  {
    run_it = SSA_steps.begin();
  }
  else
  {
    ++run_it;
    if (run_it == SSA_steps.end())
    {
      // There is in fact, nothing to do
      return;
    }
  }

  // Pre-register address-of'd string/array literals so int-to-ptr casts see
  // them regardless of source-level declaration order (see
  // pre_register_addresses for scope).
  pre_register_addresses(conv, run_it, SSA_steps.end());

  // Now iterate from the start insn to convert, to the end of the list.
  for (; run_it != SSA_steps.end(); ++run_it)
    ::convert_internal_step(
      ns,
      ssa_trace,
      ssa_smt_trace,
      output_count,
      conv,
      solver_state->states.back(),
      *run_it,
      assertion_modet::Violated);

  --run_it;
  cvt_progress = run_it;
}

void runtime_encoded_equationt::push_ctx()
{
  flush_latest_instructions();

  // And push everything back.
  solver_state->states.push_back(solver_state->states.back());
  scoped_end_points.push_back(cvt_progress);
  conv.push_ctx();
}

void runtime_encoded_equationt::pop_ctx()
{
  SSA_stepst::iterator it = scoped_end_points.back();
  cvt_progress = it;

  if (SSA_steps.size() != 0)
    ++it;

  SSA_steps.erase(it, SSA_steps.end());

  conv.pop_ctx();
  scoped_end_points.pop_back();
  solver_state->states.pop_back();
}

void runtime_encoded_equationt::convert(
  smt_convt &smt_conv,
  assertion_modet mode)
{
  // The incremental path doesn't re-walk SSA_steps, so the per-assertion
  // path-assumption rewrite that vacuity mode needs cannot be applied here.
  // Fail loudly rather than producing normal-mode results under a vacuity
  // probe.
  (void)mode;
  assert(
    mode == assertion_modet::Violated &&
    "runtime_encoded_equationt::convert supports only the default mode");

  // Don't actually convert. We've already done most of the conversion by now
  // (probably), instead flush all unconverted instructions. We don't push
  // a context, because a) where do we unpop it, but b) we're never going to
  // build anything on top of this, so there's no gain by pushing it.
  flush_latest_instructions();

  // Finally, we also want to assert the set of assertions.
  if (!solver_state->states.back().assertions.empty())
    smt_conv.assert_expr(disjunction(solver_state->states.back().assertions));
}

std::shared_ptr<symex_targett> runtime_encoded_equationt::clone() const
{
  // Only permit cloning at the start of a run - there should never be any data
  // in this formula when it happens. Cloning needs to be supported so that a
  // reachability_treet can take a template equation and clone it ever time it
  // sets up a new exploration.
  assert(
    SSA_steps.size() == 0 &&
    "runtime_encoded_equationt shouldn't be "
    "cloned when it contains data");
  return std::shared_ptr<symex_targett>(
    new runtime_encoded_equationt(ns, conv));
}

tvt runtime_encoded_equationt::ask_solver_question(const expr2tc &question)
{
  tvt final_res;

  // So - we have a formula, we want to work out whether it's true, false, or
  // unknown. Before doing anything, first push a context, as we'll need to
  // wipe some state afterwards.
  push_ctx();

  // Convert the question (must be a bool) at this outer context level so its
  // AST is cached above the two probe scopes below; otherwise each probe's
  // pop_ctx would drop the cache entry and force a re-conversion.
  assert(is_bool_type(question));
  conv.convert_ast(question);

  // The proposition also needs to be guarded with the in-program assumptions,
  // which are not necessarily going to be part of the state guard.
  conv.assert_expr(solver_state->states.back().assumpt_expr);

  // Now, how to ask the question? We make an assertion, check it, pop it, then
  // check another.
  // Those assertions are just is-the-prop-true, is-the-prop-false. Valid
  // results are true, false, both.
  push_ctx();
  conv.assert_expr(question);
  smt_resultt res1 = conv.dec_solve();
  pop_ctx();
  push_ctx();
  conv.assert_expr(not2tc(question));
  smt_resultt res2 = conv.dec_solve();
  pop_ctx();

  // So; which result?
  if (
    res1 == P_ERROR || res1 == P_SMTLIB || res2 == P_ERROR || res2 == P_SMTLIB)
  {
    log_error("Solver returned error while asking question");
    abort();
  }
  else if (res1 == P_SATISFIABLE && res2 == P_SATISFIABLE)
  {
    // Both ways are satisfiable; result is unknown.
    final_res = tvt(tvt::TV_UNKNOWN);
  }
  else if (res1 == P_SATISFIABLE && res2 == P_UNSATISFIABLE)
  {
    // Truth of question is satisfiable; other not; so we're true.
    final_res = tvt(tvt::TV_TRUE);
  }
  else if (res1 == P_UNSATISFIABLE && res2 == P_SATISFIABLE)
  {
    // Truth is unsat, false is sat, proposition is false
    final_res = tvt(tvt::TV_FALSE);
  }
  else
  {
    pop_ctx();
    throw dual_unsat_exception();
  }

  // We have our result; pop off the questions / formula we've asked.
  pop_ctx();

  return final_res;
}
