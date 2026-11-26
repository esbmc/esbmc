#include <cassert>
#include <goto-symex/trace/build_goto_trace.h>
#include <goto-symex/equation/slice.h>
#include <goto-symex/state/renaming.h>
#include <goto-symex/witness/witnesses.h>
#include <solvers/smt/smt_conv.h>
#include <map>
#include <unordered_map>
#include <unordered_set>

static bool reads_through_dereference(const expr2tc &e);

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
    // Symex renames this index but leaves its dereference in place, which the
    // solver cannot evaluate.
    expr2tc new_value = reads_through_dereference(index.index)
                          ? index.index
                          : smt_conv.get(index.index);
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

/* Whether the model produced @p e rather than symex propagating it:
 * is_constant_expr is syntactic and admits aggregates with symbolic
 * elements. */
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

expr2tc build_rhs(smt_convt &smt_conv, const expr2tc &rhs)
{
  if (is_nil_expr(rhs) || is_model_value(rhs))
    return rhs;

  auto new_rhs = smt_conv.get(rhs);
  renaming::renaming_levelt::get_original_name(
    new_rhs, symbol_renaming_level::level0);
  return new_rhs;
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

  /* Equal types rule out a sub-object target (`&a[3]`, `&w.s`), which would
   * leave @p ssa_lhs naming the enclosing object: no type contains a distinct
   * sub-object of its own type. */
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
    if (
      is_nil_expr(src) || !is_array_type(src->type) ||
      reads_through_dereference(index.index))
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

/* The value of the lvalue an assignment step prints, read as that lvalue
 * rather than as the whole object the SSA assignment rewrote: symex lowers
 * `s.f = v` into `s = s WITH [f := v]`, so evaluating the RHS answers for all
 * of `s`, at one model query per leaf and printed in full on every step that
 * touches it. Nil where the model does not pin the component down; the caller
 * then falls back to the RHS. */
static expr2tc build_component_value(
  smt_convt &smt_conv,
  const symex_target_equationt::SSA_stept &step)
{
  if (
    is_nil_expr(step.lhs) || is_nil_expr(step.original_lhs) ||
    !is_symbol2t(step.lhs) || is_symbol2t(step.original_lhs))
    return expr2tc();

  expr2tc component = rebase_on_ssa_lhs(smt_conv, step.original_lhs, step.lhs);
  if (is_nil_expr(component))
    return expr2tc();

  expr2tc value = build_rhs(smt_conv, component);
  return is_model_value(value) ? value : expr2tc();
}

/* The component the step prints where that can be read on its own, the whole
 * rewritten object otherwise. */
static expr2tc build_assignment_value(
  smt_convt &smt_conv,
  const symex_target_equationt::SSA_stept &step)
{
  expr2tc component = build_component_value(smt_conv, step);
  if (!is_nil_expr(component))
    return component;

  return build_rhs(
    smt_conv, is_nil_expr(step.original_rhs) ? step.rhs : step.original_rhs);
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

/* The instruction of the step each nondet symbol first appears in; null once
 * a waypoint names the symbol. */
using nondet_originst =
  std::unordered_map<irep_idt, const goto_programt::instructiont *>;

/* Record what a __VERIFIER_nondet_*() call returned on this path: the nondet
 * symbol its `lhs = NONDET` lowering assigns, not the whole lhs object. */
static void record_nondet_return(
  smt_convt &smt_conv,
  const symex_target_equationt::SSA_stept &step,
  bool ignored,
  std::list<goto_trace_stept::nondet_returnt> &out,
  nondet_originst &origin)
{
  if (!step.is_assignment())
    return;

  // A store into an initialised array or struct is folded into the aggregate,
  // which also holds earlier calls' values (#10770). A symbol belongs to the
  // instruction of the step it first appears in, so a later call never takes
  // it. A store through a pointer with several targets writes one call's
  // symbol in one step per target, all at the same instruction.
  const goto_programt::instructiont *const pc = &*step.source.pc;
  expr2tc nondet;
  bool ambiguous = false;
  for (const expr2tc &symbol : symex_slicet::get_nondet_symbols(step.rhs))
  {
    if (origin.emplace(to_symbol2t(symbol).thename, pc).first->second != pc)
      continue;
    // Two of this instruction's symbols: which visit each came from is lost.
    if (nondet && to_symbol2t(nondet).thename != to_symbol2t(symbol).thename)
      ambiguous = true;
    nondet = symbol;
  }
  if (!nondet || ambiguous || ignored)
    return;

  // Other nondet assignments (an uninitialised local, a contract's havoc of a
  // call's result) are not at a nondet call in the source.
  const auto rparen = nondet_call_rparen(step.source.pc->location);
  if (!rparen || !smt_conv.l_get(step.guard).is_true())
    return;

  // get() yields nil for a value it cannot convert, e.g. an --ir-ieee long
  // double.
  expr2tc value = smt_conv.get(nondet);
  if (!is_nil_expr(value) && is_constant_expr(value))
  {
    out.push_back({*rparen, value, step.source.thread_nr});
    origin[to_symbol2t(nondet).thename] = nullptr;
  }
}

/* Record a thread spawned on this path. The spawn intrinsic assigns the new
 * thread's number to the call's result in a hidden step, after run_intrinsic
 * has moved the pc past the call. */
static void record_spawn(
  smt_convt &smt_conv,
  const symex_target_equationt::SSA_stept &step,
  size_t nondet_returns_before,
  std::list<goto_trace_stept::spawnt> &out)
{
  if (
    !step.is_assignment() || !step.source.prog ||
    step.source.pc == step.source.prog->instructions.begin() ||
    !is_constant_int2t(step.rhs))
    return;

  const goto_programt::const_targett call = std::prev(step.source.pc);
  if (!call->is_function_call())
    return;
  const code_function_call2t &code = to_code_function_call2t(call->code);
  if (
    !is_symbol2t(code.function) ||
    to_symbol2t(code.function).thename != "c:@F@__ESBMC_spawn_thread" ||
    !is_symbol2t(step.original_lhs) || !is_symbol2t(code.ret) ||
    to_symbol2t(step.original_lhs).thename != to_symbol2t(code.ret).thename ||
    !smt_conv.l_get(step.guard).is_true())
    return;

  out.push_back(
    {static_cast<unsigned>(to_constant_int2t(step.rhs).value.to_uint64()),
     step.source.thread_nr,
     {call, step.stack_trace()},
     nondet_returns_before});
}

/* Tracks which SSA symbols were computed from constants alone. A branch whose
 * condition reads only such symbols is fixed by the program text, so a witness
 * need not name its direction. */
struct input_trackert
{
  std::unordered_set<std::string> computed;

  /* Whether @p e reads a nondet value or a symbol no earlier assignment
   * computed from constants. */
  bool reads_input(const expr2tc &e) const
  {
    if (is_nil_expr(e))
      return false;
    if (is_sideeffect2t(e))
      return true;
    if (is_symbol2t(e))
      return find_nondet_in_expr(e) ||
             !computed.count(to_symbol2t(e).get_symbol_name());

    bool found = false;
    e->foreach_operand(
      [&](const expr2tc &op) { found = found || reads_input(op); });
    return found;
  }

  void note_assignment(const symex_target_equationt::SSA_stept &step)
  {
    if (step.is_assignment() && is_symbol2t(step.lhs) && !reads_input(step.rhs))
      computed.insert(to_symbol2t(step.lhs).get_symbol_name());
  }

  void annotate_branch(
    const symex_target_equationt::SSA_stept &step,
    goto_trace_stept &trace_step) const
  {
    if (step.is_branching())
      trace_step.nondet_dependent = reads_input(step.cond);
  }
};

static bool is_races_flag(const expr2tc &e)
{
  return is_symbol2t(e) && to_symbol2t(e).thename == "c:@F@__ESBMC_races_flag";
}

/* The slot index of the first __ESBMC_races_flag read in @p e, or nil. @p seen
 * memoises shared nodes: an SSA condition is a DAG, and walking it as a tree
 * is exponential (regression/esbmc/nested_array_no_simplify_scale_fail). */
static expr2tc
races_flag_read_key(const expr2tc &e, std::unordered_set<const expr2t *> &seen)
{
  if (!seen.insert(e.get()).second)
    return expr2tc();

  if (is_index2t(e) && is_races_flag(to_index2t(e).source_value))
    return to_index2t(e).index;

  expr2tc key;
  e->foreach_operand([&key, &seen](const expr2tc &op) {
    if (is_nil_expr(key) && !is_nil_expr(op))
      key = races_flag_read_key(op, seen);
  });
  return key;
}

/* Tracks the latest write to each __ESBMC_races_flag slot, keyed by the
 * (object, offset) value the slot index takes in the model, so a violated race
 * assert can name the access that set the slot it reads (#9688). The writes
 * are hidden SSA steps, so they are tracked before hidden steps are dropped. */
struct race_flag_trackert
{
  struct slot_writet
  {
    expr2tc key;
    unsigned thread_nr;
    bool set;
    goto_trace_stept::accesst access;
  };
  std::vector<slot_writet> writes;

  void note_assignment(
    smt_convt &smt_conv,
    const symex_target_equationt::SSA_stept &step,
    bool ignored)
  {
    if (
      !step.is_assignment() || !is_races_flag(step.lhs) ||
      !is_with2t(step.rhs) || ignored || !smt_conv.l_get(step.guard).is_true())
      return;

    const with2t &with = to_with2t(step.rhs);
    slot_writet write{
      smt_conv.get_by_ast(with.update_field),
      step.source.thread_nr,
      smt_conv.l_get(with.update_value).is_true(),
      {step.source.pc, step.stack_trace()}};

    for (slot_writet &w : writes)
      if (w.key == write.key)
      {
        w = std::move(write);
        return;
      }
    writes.push_back(std::move(write));
  }

  void annotate_violation(
    smt_convt &smt_conv,
    const symex_target_equationt::SSA_stept &step,
    goto_trace_stept &trace_step) const
  {
    if (trace_step.guard)
      return;

    std::unordered_set<const expr2t *> seen;
    const expr2tc key = races_flag_read_key(step.cond, seen);
    if (is_nil_expr(key))
      return;

    const expr2tc value = smt_conv.get_by_ast(key);
    for (const slot_writet &w : writes)
      if (w.key == value && w.set && w.thread_nr != trace_step.thread_nr)
        trace_step.racing_access = w.access;
  }
};

/* The conditional gotos that share their column with another in a run of
 * instructions on one line: a && or || condition becomes one goto per operand,
 * all at the statement's location. goto-convert marks the operand gotos it
 * makes, but not the goto that then tests the lowered condition's value. */
static std::unordered_set<const goto_programt::instructiont *>
condition_operand_gotos(const goto_programt &prog)
{
  std::unordered_set<const goto_programt::instructiont *> operands;
  std::map<irep_idt, std::vector<const goto_programt::instructiont *>> run;
  auto end_run = [&]() {
    for (const auto &[column, gotos] : run)
      if (gotos.size() > 1)
        operands.insert(gotos.begin(), gotos.end());
    run.clear();
  };

  irep_idt line;
  for (const auto &instruction : prog.instructions)
  {
    if (instruction.location.get_line() != line)
    {
      end_run();
      line = instruction.location.get_line();
    }
    if (instruction.is_goto() && !is_true(instruction.guard))
      run[instruction.location.get_column()].push_back(&instruction);
    if (instruction.location.is_short_circuit_operand())
      operands.insert(&instruction);
  }
  end_run();
  return operands;
}

/// Hidden steps are internal SSA bookkeeping (e.g. phi-merge nodes at
/// control-flow joins). They carry a synthesised value and a source location
/// borrowed from a branch, so surfacing them in the trace yields
/// contradictory-looking states (see discussion #5701). They are never a
/// user-visible source assignment, so drop them regardless of slicing. A
/// --partial-order event has no value of its own; the assignment it marks is
/// already in the trace.
static bool is_bookkeeping(const symex_target_equationt::SSA_stept &step)
{
  return step.hidden || step.is_event();
}

void build_goto_trace(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  goto_tracet &goto_trace)
{
  unsigned step_nr = 0;
  input_trackert inputs;
  race_flag_trackert race_flags;
  std::unordered_map<
    const goto_programt *,
    std::unordered_set<const goto_programt::instructiont *>>
    operand_gotos;
  std::list<goto_trace_stept::nondet_returnt> nondet_returns;
  nondet_originst nondet_origin;
  std::list<goto_trace_stept::spawnt> spawns;

  // l_get() memoises against the current model internally (the cache is
  // cleared on every solve / context change), so the thousands of repeated
  // guard-AST queries this loop issues collapse to one solver call each
  // without any explicit scope management here.
  auto record =
    [&](const symex_target_equationt::SSA_stept &step, bool ignored) {
      inputs.note_assignment(step);
      race_flags.note_assignment(smt_conv, step, ignored);
      record_nondet_return(
        smt_conv, step, ignored, nondet_returns, nondet_origin);
      record_spawn(smt_conv, step, nondet_returns.size(), spawns);
    };
  auto trace_step = [&](const symex_target_equationt::SSA_stept &SSA_step) {
    record(SSA_step, SSA_step.ignore);

    if (is_bookkeeping(SSA_step))
      return;

    // is_true() also drops steps whose guard the solver could not evaluate:
    // such a step has no authentic state to report (see #6191).
    if (SSA_step.ignore || !smt_conv.l_get(SSA_step.guard).is_true())
      return;

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
        goto_trace_step.value = build_assignment_value(smt_conv, SSA_step);

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
        return;
      }
      catch (const array_type2t::dyn_sized_array_excp &e)
      {
        log_debug(
          "trace",
          "skipping assignment at {} (symbolic-size array, e.g. argv)",
          SSA_step.source.pc->location.as_string());
        return;
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
      race_flags.annotate_violation(smt_conv, SSA_step, goto_trace_step);
    }
    // Keeps the opposite idiom on purpose: here guard is a direction bit, not
    // a violation flag, so unknown has no fail-safe value and flipping would
    // swap one invented branch direction for another.
    else if (SSA_step.is_assume() || SSA_step.is_branching())
      goto_trace_step.guard = !smt_conv.l_get(SSA_step.cond).is_false();

    inputs.annotate_branch(SSA_step, goto_trace_step);
    if (SSA_step.is_branching() && SSA_step.source.prog)
    {
      auto [it, fresh] = operand_gotos.try_emplace(SSA_step.source.prog);
      if (fresh)
        it->second = condition_operand_gotos(*SSA_step.source.prog);
      goto_trace_step.condition_operand =
        it->second.count(&*SSA_step.source.pc);
    }
    goto_trace_step.nondet_returns.swap(nondet_returns);
    goto_trace_step.spawns.swap(spawns);

    goto_trace.steps.push_back(goto_trace_step);
  };
  target.for_each_step(
    trace_step,
    [&](const symex_target_equationt::SSA_stept &step) { record(step, true); });
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
