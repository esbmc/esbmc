#include <goto-symex/incremental_algorithms.h>
#include <irep2/irep2_utils.h>
#include <util/i2string.h>
#include <util/message.h>

void incremental_smt_algorithm::init()
{
  assumpt_ast = conv->convert_ast(gen_true_expr());
  assertions.clear();
  ignored_count = 0;
  output_count = 0;
}

bool incremental_smt_algorithm::run(SSA_stepst &steps)
{
  init();

  for (auto &step : steps)
  {
    if (step.ignore)
    {
      ignored_count += 1;
      continue;
    }
    run_on_step(step);
  }

  if (!assertions.empty())
    conv->assert_ast(conv->make_n_ary_or(assertions));

  return true;
}

smt_convt::resultt incremental_smt_algorithm::solve()
{
  conv->pre_solve();
  return conv->dec_solve();
}

tvt incremental_smt_algorithm::step_online(SSA_stept &step)
{
  if (step.ignore)
  {
    ignored_count += 1;
    return tvt(tvt::TV_UNKNOWN);
  }

  // Assertions are checked immediately via push/pop, then their condition is
  // committed to the path (same as an assumption) so subsequent steps see it.
  if (step.is_assert() || step.is_assume())
  {
    tvt result = ask_solver_question(step.cond);
    step.guard_ast = conv->convert_ast(step.guard);
    step.cond_ast = conv->convert_ast(step.cond);
    assumpt_ast = conv->mk_and(assumpt_ast, step.cond_ast);

    // TODO: TV_FALSE means the path is dead; signal the caller to stop
    //       exploring this path

    // TODO: TV_UNKNOWN for assertions means that is SAT in some conditions.

    // TODO: TV_UNKNOWN assertions in multi-property should be a assume.

    return result;
  }

  run_on_step(step);
  return tvt(tvt::TV_UNKNOWN);
}

BigInt incremental_smt_algorithm::ignored() const
{
  return ignored_count;
}

void incremental_smt_algorithm::run_on_assignment(SSA_stept &step)
{
  step.guard_ast = conv->convert_ast(step.guard);
  conv->convert_assign(step.cond);
}

void incremental_smt_algorithm::run_on_assume(SSA_stept &step)
{
  step.guard_ast = conv->convert_ast(step.guard);
  step.cond_ast = conv->convert_ast(step.cond);
  assumpt_ast = conv->mk_and(assumpt_ast, step.cond_ast);
}

void incremental_smt_algorithm::run_on_assert(SSA_stept &step)
{
  step.guard_ast = conv->convert_ast(step.guard);
  step.cond_ast = conv->convert_ast(step.cond);
  step.cond_ast = conv->imply_ast(assumpt_ast, step.cond_ast);
  assertions.push_back(conv->invert_ast(step.cond_ast));
}

void incremental_smt_algorithm::run_on_output(SSA_stept &step)
{
  for (const expr2tc &arg : step.output_args)
  {
    if (is_constant_expr(arg) || is_constant_string2t(arg))
    {
      step.converted_output_args.push_back(arg);
    }
    else
    {
      expr2tc sym =
        symbol2tc(arg->type, "symex::output::" + i2string(output_count++));
      expr2tc eq = equality2tc(sym, arg);
      conv->convert_assign(eq);
      step.converted_output_args.push_back(sym);
    }
  }
}

void incremental_smt_algorithm::run_on_renumber(SSA_stept &step)
{
  conv->renumber_symbol_address(step.guard, step.lhs, step.rhs);
}

void incremental_smt_algorithm::run_on_branching(SSA_stept &step)
{
  step.guard_ast = conv->convert_ast(step.guard);
  step.cond_ast = conv->convert_ast(step.cond);
}

tvt incremental_smt_algorithm::ask_solver_question(const expr2tc &question)
{
  assert(is_bool_type(question));

  // All temporary assertions are scoped under one outer push so the base
  // context is intact when we return.
  conv->push_ctx();
  smt_astt q = conv->convert_ast(question);
  conv->assert_ast(assumpt_ast);

  // Check ¬q first.  UNSAT means q is always true → TV_TRUE in one call,
  // which is the common case on a live, well-constrained path.  Only when
  // ¬q is SAT do we make a second call to distinguish TV_FALSE (q can never
  // hold) from TV_UNKNOWN (both q and ¬q are satisfiable).
  conv->push_ctx();
  conv->assert_ast(conv->invert_ast(q));
  smt_convt::resultt res_neg = conv->dec_solve();
  conv->pop_ctx();

  if (res_neg == smt_convt::P_ERROR || res_neg == smt_convt::P_SMTLIB)
  {
    conv->pop_ctx();
    log_error("Solver returned error while asking question");
    abort();
  }

  if (res_neg == smt_convt::P_UNSATISFIABLE)
  {
    conv->pop_ctx();
    return tvt(tvt::TV_TRUE);
  }

  // ¬q is satisfiable; now check q to tell TV_FALSE from TV_UNKNOWN.
  conv->push_ctx();
  conv->assert_ast(q);
  smt_convt::resultt res_pos = conv->dec_solve();
  conv->pop_ctx();

  conv->pop_ctx();

  if (res_pos == smt_convt::P_ERROR || res_pos == smt_convt::P_SMTLIB)
  {
    log_error("Solver returned error while asking question");
    abort();
  }

  if (res_pos == smt_convt::P_UNSATISFIABLE)
    return tvt(tvt::TV_FALSE);

  return tvt(tvt::TV_UNKNOWN);
}
