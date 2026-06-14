#include <cassert>
#include <functional>
#include <goto-symex/goto_symex.h>
#include <goto-symex/goto_symex_state.h>
#include <goto-symex/symex_target_equation.h>
#include <langapi/language_util.h>
#include <solvers/smt/smt_conv.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/std_expr.h>

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
  std::function<void(const expr2tc &)> walk = [&](const expr2tc &e) {
    if (!e)
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

void convert_internal_step(
  const namespacet &ns,
  bool ssa_trace,
  bool ssa_smt_trace,
  unsigned &output_count,
  smt_convt &smt_conv,
  equation_conversion_statet &state,
  symex_target_equationt::SSA_stept &step,
  bool vacuity_mode)
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

  smt_conv.convert_ast(step.guard);

  if (step.is_assume() || step.is_assert() || step.is_branching())
  {
    smt_conv.convert_ast(step.cond);
    if (ssa_smt_trace)
      smt_conv.dump_expr(step.cond);
  }
  else if (step.is_assignment())
  {
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

  if (step.is_assert())
  {
    if (vacuity_mode)
    {
      // Vacuity probe: ask whether the path to this claim is reachable at
      // all, ignoring the claim itself. If the OR of all kept claims'
      // path assumption is UNSAT, every discharge was vacuous.
      step.cond_expr = state.assumpt_expr;
      state.assertions.push_back(state.assumpt_expr);
    }
    else
    {
      step.cond_expr = implies2tc(state.assumpt_expr, step.cond);
      state.assertions.push_back(not2tc(step.cond_expr));
    }
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
  SSA_step.type = goto_trace_stept::ASSIGNMENT;
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
  SSA_step.type = goto_trace_stept::OUTPUT;
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
  SSA_step.type = goto_trace_stept::BREANCHING;
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
  SSA_step.type = goto_trace_stept::ASSUME;
  SSA_step.source = source;
  SSA_step.loop_number = loop_number;

  if (debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::assertion(
  const expr2tc &guard,
  const expr2tc &cond,
  const std::string &msg,
  std::vector<stack_framet> stack_trace,
  const sourcet &source,
  unsigned loop_number)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type = goto_trace_stept::ASSERT;
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
  SSA_step.type = goto_trace_stept::RENUMBER;
  SSA_step.source = source;

  if (debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::convert(smt_convt &smt_conv, bool vacuity_mode)
{
  // Register address-taken objects first so int-to-ptr casts see the full
  // set of candidate objects regardless of source-level declaration order.
  pre_register_addresses(smt_conv, SSA_steps.begin(), SSA_steps.end());

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
      vacuity_mode);

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
  case goto_trace_stept::ASSERT:
    out << "ASSERT"
        << "\n";
    break;
  case goto_trace_stept::ASSUME:
    out << "ASSUME"
        << "\n";
    break;
  case goto_trace_stept::OUTPUT:
    out << "OUTPUT"
        << "\n";
    break;
  case goto_trace_stept::BREANCHING:
    out << "BRANCHING"
        << "\n";
    break;
  case goto_trace_stept::ASSIGNMENT:
    out << "ASSIGNMENT (";
    out << (hidden ? "HIDDEN" : "") << ")\n";
    break;

  default:
    assert(
      type == goto_trace_stept::SKIP && config.options.get_bool_option("ltl"));
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

void symex_target_equationt::check_for_duplicate_assigns() const
{
  std::map<std::string, unsigned int> countmap;
  unsigned int i = 0;

  for (const auto &SSA_step : SSA_steps)
  {
    i++;
    if (!SSA_step.is_assignment())
      continue;

    const equality2t &ref = to_equality2t(SSA_step.cond);
    const symbol2t &sym = to_symbol2t(ref.side_1);
    countmap[sym.get_symbol_name()]++;
  }

  for (std::map<std::string, unsigned int>::const_iterator it =
         countmap.begin();
       it != countmap.end();
       ++it)
  {
    if (it->second != 1)
    {
      log_status("Symbol \"{}\" appears {} times", it->first, it->second);
    }
  }

  log_status("Checked {} insns", i);
}

unsigned int symex_target_equationt::clear_assertions()
{
  unsigned int num_asserts = 0;

  for (SSA_stepst::iterator it = SSA_steps.begin(); it != SSA_steps.end(); ++it)
  {
    if (it->type == goto_trace_stept::ASSERT)
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

    // Just roll on
  }

  // Register address-taken objects first so int-to-ptr casts see the full
  // set of candidate objects regardless of source-level declaration order.
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
      /*vacuity_mode=*/false);

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

void runtime_encoded_equationt::convert(smt_convt &smt_conv, bool vacuity_mode)
{
  // The incremental path doesn't re-walk SSA_steps, so the per-assertion
  // path-assumption rewrite that vacuity mode needs cannot be applied here.
  // Fail loudly rather than producing normal-mode results under a vacuity
  // probe.
  (void)vacuity_mode;
  assert(
    !vacuity_mode &&
    "runtime_encoded_equationt::convert does not support vacuity mode");

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
