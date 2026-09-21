#include <cassert>
#include <goto-symex/trace/build_goto_trace.h>
#include <goto-symex/state/renaming.h>
#include <goto-symex/witness/witnesses.h>
#include <solvers/smt/smt_conv.h>

expr2tc build_lhs(smt_convt &smt_conv, const expr2tc &lhs)
{
  if (is_nil_expr(lhs))
    return lhs;

  expr2tc new_lhs = lhs;
  switch (new_lhs->expr_id)
  {
  case expr2t::index_id:
  {
    index2t index = to_index2t(new_lhs);

    // Build new source value, it might be an index, in case of
    // multidimensional arrays
    expr2tc new_source_value = build_lhs(smt_conv, index.source_value);
    expr2tc new_value = smt_conv.get(index.index);
    new_lhs = index2tc(new_lhs->type, new_source_value, new_value);
    break;
  }

  case expr2t::typecast_id:
    new_lhs = to_typecast2t(new_lhs).from;
    break;

  case expr2t::bitcast_id:
    new_lhs = to_bitcast2t(new_lhs).from;
    break;

  default:
    break;
  }

  renaming::renaming_levelt::get_original_name(
    new_lhs, symbol_renaming_level::level0);
  return new_lhs;
}

expr2tc build_rhs(smt_convt &smt_conv, const expr2tc &rhs)
{
  if (is_nil_expr(rhs) || is_constant_expr(rhs))
    return rhs;

  auto new_rhs = smt_conv.get(rhs);
  renaming::renaming_levelt::get_original_name(
    new_rhs, symbol_renaming_level::level0);
  return new_rhs;
}

/* Whether @p e is something the model produced rather than something symex
 * propagated. `is_constant_expr` cannot answer this: it is syntactic, and
 * constant propagation leaves `constant_struct`/`constant_array` nodes whose
 * elements are still symbolic. */
static bool is_model_value(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;

  if (is_address_of2t(e))
    return true;

  if (is_symbol2t(e))
    return to_symbol2t(e).thename == "NULL";

  if (!is_constant_expr(e))
    return false;

  bool resolved = true;
  e->foreach_operand([&resolved](const expr2tc &op) {
    if (!is_model_value(op))
      resolved = false;
  });
  return resolved;
}

/* Rebuild the source lvalue @p lvalue over @p ssa_lhs, the SSA symbol holding
 * the object's post-assignment value. Nil unless the lvalue is a chain of
 * member and constant-index accesses rooted at that same object. */
static expr2tc rebase_on_ssa_lhs(
  smt_convt &smt_conv,
  const expr2tc &lvalue,
  const expr2tc &ssa_lhs)
{
  switch (lvalue->expr_id)
  {
  case expr2t::symbol_id:
  {
    expr2tc root = ssa_lhs;
    renaming::renaming_levelt::get_original_name(
      root, symbol_renaming_level::level0);
    if (
      !is_symbol2t(root) ||
      to_symbol2t(root).thename != to_symbol2t(lvalue).thename)
      return expr2tc();
    return ssa_lhs;
  }

  /* symex resolved the dereference to the object being assigned before it
   * lowered the write, so the accesses above it in the source lvalue are
   * accesses into that object -- but only where the dereference covered the
   * whole of it. A sub-object target (`&a[3]`, `&w.s`) leaves @p ssa_lhs
   * naming the enclosing object, and equal types rule that out: no type
   * contains a distinct sub-object of its own type. */
  case expr2t::dereference_id:
    return lvalue->type == ssa_lhs->type ? ssa_lhs : expr2tc();

  case expr2t::member_id:
  {
    const member2t &mem = to_member2t(lvalue);
    expr2tc src = rebase_on_ssa_lhs(smt_conv, mem.source_value, ssa_lhs);
    if (is_nil_expr(src) || !is_structure_type(src->type))
      return expr2tc();

    std::optional<unsigned int> nr =
      struct_union_get_component_number(src->type, mem.member);
    if (!nr)
      return expr2tc();

    return member2tc(struct_union_members(src->type)[*nr], src, mem.member);
  }

  case expr2t::index_id:
  {
    const index2t &index = to_index2t(lvalue);
    expr2tc src = rebase_on_ssa_lhs(smt_conv, index.source_value, ssa_lhs);
    if (is_nil_expr(src) || !is_array_type(src->type))
      return expr2tc();

    expr2tc idx = smt_conv.get(index.index);
    if (is_nil_expr(idx) || !is_constant_int2t(idx))
      return expr2tc();

    return index2tc(to_array_type(src->type).subtype, src, idx);
  }

  default:
    return expr2tc();
  }
}

/* The value of the lvalue an assignment step prints, read from the model as
 * that lvalue rather than as the whole object the SSA assignment rewrote.
 *
 * symex lowers a write to a component -- `s.f = v`, `a[i] = v` -- into a
 * whole-object update `s = s WITH [f := v]`, so evaluating the step's RHS
 * answers for all of `s`: a value the printed lvalue does not have, obtained
 * at one model query per leaf of the object and printed in full, on every
 * step that touches it. Over a loop writing into a large aggregate both costs
 * grow quadratically in the number of iterations.
 *
 * Nil when the lvalue is not such a component, or when the model does not pin
 * it down; the caller then falls back to evaluating the RHS. */
static expr2tc build_component_value(
  smt_convt &smt_conv,
  const symex_target_equationt::SSA_stept &step)
{
  if (
    !is_symbol2t(step.lhs) || is_nil_expr(step.original_lhs) ||
    is_symbol2t(step.original_lhs))
    return expr2tc();

  expr2tc component = rebase_on_ssa_lhs(smt_conv, step.original_lhs, step.lhs);
  if (is_nil_expr(component))
    return expr2tc();

  expr2tc value = build_rhs(smt_conv, component);
  return is_model_value(value) ? value : expr2tc();
}

/* The claim a violated assert reports is written in source terms -- the GOTO
 * guard still says `d->devnum` -- while the value only exists in the SSA
 * condition symex built from it, where dereference lowering has replaced the
 * pointer read with the object it resolved to. The two stay structurally
 * parallel down to that substitution, so walking them together pairs each
 * source lvalue with something the model can evaluate. Where the shapes
 * diverge for any other reason -- a dereference that resolved to several
 * objects, a byte-level access -- the walk stops and that operand is simply
 * not recorded, which costs replay precision and nothing else (#7858).
 *
 * Only reads with no assignment step of their own are worth recording; every
 * other value in the claim already reaches the witness through
 * get_formated_assignment(). Those are the ones rooted at a dereference. */
static bool reads_through_dereference(const expr2tc &e)
{
  if (is_dereference2t(e))
    return true;

  bool found = false;
  e->foreach_operand([&found](const expr2tc &op) {
    if (!is_nil_expr(op) && reads_through_dereference(op))
      found = true;
  });
  return found;
}

/* The value of @p source in the counterexample, when it is a read this trace
 * records nowhere else: a scalar lvalue reached through a dereference. Nil
 * when it is anything else, or when the model does not pin it down. */
static expr2tc nondet_read_value(
  smt_convt &smt_conv,
  const expr2tc &source,
  const expr2tc &renamed)
{
  if (!is_member2t(source) && !is_index2t(source))
    return expr2tc();

  if (
    !reads_through_dereference(source) || !is_scalar_type(source->type) ||
    source->type != renamed->type)
    return expr2tc();

  expr2tc value = smt_conv.get(renamed);
  return is_constant_expr(value) ? value : expr2tc();
}

static void collect_nondet_reads(
  smt_convt &smt_conv,
  const expr2tc &source,
  const expr2tc &renamed_in,
  std::list<std::pair<expr2tc, expr2tc>> &out)
{
  if (is_nil_expr(source) || is_nil_expr(renamed_in))
    return;

  /* An assert's SSA condition is the claim under its path guards, so strip
   * those implications to reach the part the GOTO guard corresponds to. A
   * claim that is itself an implication keeps its own shape. */
  expr2tc renamed = renamed_in;
  while (is_implies2t(renamed) && !is_implies2t(source))
    renamed = to_implies2t(renamed).side_2;

  expr2tc value = nondet_read_value(smt_conv, source, renamed);
  if (!is_nil_expr(value))
  {
    out.emplace_back(source, value);
    return;
  }

  /* Descend only while the substitution is the single difference between the
   * two: a differing kind or arity means the shapes have parted and the
   * operands no longer correspond. */
  if (source->expr_id != renamed->expr_id)
    return;

  std::vector<expr2tc> source_ops, renamed_ops;
  source->foreach_operand(
    [&source_ops](const expr2tc &op) { source_ops.push_back(op); });
  renamed->foreach_operand(
    [&renamed_ops](const expr2tc &op) { renamed_ops.push_back(op); });

  if (source_ops.size() != renamed_ops.size())
    return;

  for (size_t i = 0; i < source_ops.size(); i++)
    collect_nondet_reads(smt_conv, source_ops[i], renamed_ops[i], out);
}

/* Pair the claim of a violated assert with the values its dereference-rooted
 * reads take here. A claim that held records nothing. */
static void record_violated_reads(
  smt_convt &smt_conv,
  const expr2tc &ssa_cond,
  goto_trace_stept &step)
{
  if (step.guard)
    return;

  collect_nondet_reads(smt_conv, step.pc->guard, ssa_cond, step.nondet_reads);
}

void build_goto_trace(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  goto_tracet &goto_trace)
{
  unsigned step_nr = 0;

  // l_get() memoises against the current model internally (the cache is
  // cleared on every solve / context change), so the thousands of repeated
  // guard-AST queries this loop issues collapse to one solver call each
  // without any explicit scope management here.
  for (auto const &SSA_step : target.SSA_steps)
  {
    // Hidden steps are internal SSA bookkeeping (e.g. phi-merge nodes at
    // control-flow joins). They carry a synthesised value and a source
    // location borrowed from a branch, so surfacing them in the trace yields
    // contradictory-looking states (see discussion #5701). They are never a
    // user-visible source assignment, so drop them regardless of slicing.
    if (SSA_step.hidden)
      continue;

    // is_true() also drops steps whose guard the solver could not evaluate:
    // such a step has no authentic state to report (see #6191).
    if (SSA_step.ignore || !smt_conv.l_get(SSA_step.guard).is_true())
      continue;

    goto_trace_stept goto_trace_step;

    goto_trace_step.thread_nr = SSA_step.source.thread_nr;
    goto_trace_step.pc = SSA_step.source.pc;
    goto_trace_step.comment = id2string(SSA_step.comment);
    goto_trace_step.original_lhs = SSA_step.original_lhs;
    goto_trace_step.type = SSA_step.type;
    goto_trace_step.step_nr = ++step_nr;
    if (SSA_step.output_data)
      goto_trace_step.format_string = SSA_step.output_data->format_string;

    goto_trace_step.stack_trace = SSA_step.stack_trace();

    if (SSA_step.is_assignment())
    {
      goto_trace_step.lhs = build_lhs(smt_conv, SSA_step.original_lhs);
      goto_trace_step.rhs = SSA_step.rhs;
      assert(!goto_trace_step.value);
      try
      {
        goto_trace_step.value = build_component_value(smt_conv, SSA_step);

        if (!goto_trace_step.value)
        {
          if (is_nil_expr(SSA_step.original_rhs))
            goto_trace_step.value = build_rhs(smt_conv, SSA_step.rhs);
          else
            goto_trace_step.value = build_rhs(smt_conv, SSA_step.original_rhs);
        }

        // Try asking solver if value was not built
        if (
          !goto_trace_step.value &&
          (is_unsignedbv_type(SSA_step.lhs) || is_signedbv_type(SSA_step.lhs)))
          goto_trace_step.value = smt_conv.get(SSA_step.lhs);
      }
      catch (const type2t::symbolic_type_excp &e)
      {
        log_debug(
          "trace",
          "skipping assignment at {} (symbolic type)",
          SSA_step.source.pc->location.as_string());
        continue;
      }
      catch (const array_type2t::dyn_sized_array_excp &e)
      {
        log_debug(
          "trace",
          "skipping assignment at {} (symbolic-size array, e.g. argv)",
          SSA_step.source.pc->location.as_string());
        continue;
      }
    }

    if (SSA_step.is_output() && SSA_step.output_data)
    {
      for (const auto &arg : SSA_step.output_data->converted_output_args)
      {
        if (is_constant_expr(arg))
          goto_trace_step.output_args.push_back(arg);
        else
          goto_trace_step.output_args.push_back(smt_conv.get(arg));
      }
    }

    // An unevaluatable assertion condition (e.g. one still containing a
    // quantifier) must render as violated, not as held: this is the assertion
    // the solver already reported as failing. Hence is_true(), not !is_false().
    if (SSA_step.is_assert())
    {
      goto_trace_step.guard = smt_conv.l_get(SSA_step.cond_expr).is_true();
      record_violated_reads(smt_conv, SSA_step.cond, goto_trace_step);
    }
    // Keeps the opposite idiom on purpose: here guard is a direction bit, not
    // a violation flag, so unknown has no fail-safe value and flipping would
    // swap one invented branch direction for another.
    else if (SSA_step.is_assume() || SSA_step.is_branching())
      goto_trace_step.guard = !smt_conv.l_get(SSA_step.cond).is_false();

    goto_trace.steps.push_back(goto_trace_step);
  }
}

void build_successful_goto_trace(
  const symex_target_equationt &target,
  const namespacet &ns,
  goto_tracet &goto_trace)
{
  unsigned step_nr = 0;
  for (const symex_target_equationt::SSA_stept &SSA_step : target.SSA_steps)
  {
    if (
      (SSA_step.is_assert() || SSA_step.is_assume()) &&
      (is_valid_witness_expr(ns, SSA_step.lhs)))
    {
      // When building the correctness witness, we only care about
      // asserts and assumes
      if (!(SSA_step.is_assert() || SSA_step.is_assume()))
        continue;

      goto_trace.steps.emplace_back();
      goto_trace_stept &goto_trace_step = goto_trace.steps.back();
      goto_trace_step.thread_nr = SSA_step.source.thread_nr;
      goto_trace_step.lhs = SSA_step.lhs;
      goto_trace_step.rhs = SSA_step.rhs;
      goto_trace_step.pc = SSA_step.source.pc;
      goto_trace_step.comment = id2string(SSA_step.comment);
      goto_trace_step.original_lhs = SSA_step.original_lhs;
      goto_trace_step.type = SSA_step.type;
      goto_trace_step.step_nr = step_nr++;
      if (SSA_step.output_data)
        goto_trace_step.format_string = SSA_step.output_data->format_string;
      goto_trace_step.stack_trace = SSA_step.stack_trace();
    }
  }
}
