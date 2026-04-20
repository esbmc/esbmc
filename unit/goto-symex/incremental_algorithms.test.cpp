/*******************************************************************
 Module: Tests for the incremental SMT algorithm

 Test Plan:
   Batch mode:
     - Valid assertion: solver returns UNSAT (no counterexample)
     - Violated assertion: solver returns SAT (counterexample found)
     - ask_solver_question returns correct TVT after run()

   Online mode (init + step_online):
     - Valid assertion: step_online returns TV_TRUE immediately
     - Violated assertion: step_online returns TV_FALSE immediately
     - Context persists: later steps see earlier assignments
     - Ignored steps are counted
     - ask_solver_question is callable between step_online calls
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <goto-symex/incremental_algorithms.h>
#include <goto-symex/goto_trace.h>
#include <irep2/irep2_utils.h>
#include <solvers/solve.h>
#include <util/cmdline.h>
#include <util/config.h>
#include <util/context.h>
#include <util/namespace.h>

const mode_table_et mode_table[] = {};

namespace
{

void setup_config()
{
  cmdlinet cmdline;
  config.set(cmdline);
  config.ansi_c.set_data_model(configt::LP64);
}

optionst make_options()
{
  cmdlinet cmdline;
  optionst options;
  options.cmdline(cmdline);
  options.set_option("floatbv", true);
  return options;
}

SSA_stept make_assignment(const expr2tc &lhs, const expr2tc &rhs)
{
  SSA_stept step;
  step.type = goto_trace_stept::ASSIGNMENT;
  step.guard = gen_true_expr();
  step.lhs = lhs;
  step.rhs = rhs;
  step.cond = equality2tc(lhs, rhs);
  step.ignore = false;
  step.hidden = false;
  step.loop_number = 0;
  return step;
}

SSA_stept make_assert(const expr2tc &cond, const std::string &comment = "")
{
  SSA_stept step;
  step.type = goto_trace_stept::ASSERT;
  step.guard = gen_true_expr();
  step.cond = cond;
  step.comment = comment;
  step.ignore = false;
  step.hidden = false;
  step.loop_number = 0;
  return step;
}

SSA_stept make_assume(const expr2tc &cond)
{
  SSA_stept step;
  step.type = goto_trace_stept::ASSUME;
  step.guard = gen_true_expr();
  step.cond = cond;
  step.ignore = false;
  step.hidden = false;
  step.loop_number = 0;
  return step;
}

} // namespace


SCENARIO(
  "incremental_smt_algorithm batch mode encodes SSA steps and checks validity",
  "[algorithms][z3][batch]")
{
  setup_config();
  contextt ctx;
  namespacet ns(ctx);
  optionst opts = make_options();

  type2tc uint32 = get_uint_type(32);

  GIVEN("Assignment x = 5 and valid assertion x == 5")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "batch::x1");
    expr2tc five = constant_int2tc(uint32, BigInt(5));

    SSA_stepst steps;
    steps.push_back(make_assignment(x, five));
    steps.push_back(make_assert(equality2tc(x, five)));

    algo.run(steps);

    THEN("Solver finds no counterexample (UNSAT)")
    {
      REQUIRE(algo.solve() == smt_convt::P_UNSATISFIABLE);
    }
    THEN("No steps were ignored")
    {
      REQUIRE(algo.ignored() == BigInt(0));
    }
  }

  GIVEN("Assignment x = 5 and violated assertion x == 6")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "batch::x2");
    expr2tc five = constant_int2tc(uint32, BigInt(5));
    expr2tc six = constant_int2tc(uint32, BigInt(6));

    SSA_stepst steps;
    steps.push_back(make_assignment(x, five));
    steps.push_back(make_assert(equality2tc(x, six)));

    algo.run(steps);

    THEN("Solver finds a counterexample (SAT)")
    {
      REQUIRE(algo.solve() == smt_convt::P_SATISFIABLE);
    }
  }

  GIVEN("An ignored assignment step")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "batch::x3");
    SSA_stept s = make_assignment(x, constant_int2tc(uint32, BigInt(5)));
    s.ignore = true;

    SSA_stepst steps;
    steps.push_back(s);
    algo.run(steps);

    THEN("Ignored step is counted")
    {
      REQUIRE(algo.ignored() == BigInt(1));
    }
  }

  GIVEN("Assignment x = 5 and ask_solver_question after run()")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "batch::x4");
    expr2tc five = constant_int2tc(uint32, BigInt(5));
    expr2tc six = constant_int2tc(uint32, BigInt(6));

    SSA_stepst steps;
    steps.push_back(make_assignment(x, five));
    algo.run(steps);

    THEN("x == 5 is always true")
    {
      REQUIRE(algo.ask_solver_question(equality2tc(x, five)).is_true());
    }
    THEN("x == 6 is always false")
    {
      REQUIRE(algo.ask_solver_question(equality2tc(x, six)).is_false());
    }
  }

  GIVEN("Assumption x > 3 and valid assertion x > 0")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "batch::x5");
    expr2tc zero = constant_int2tc(uint32, BigInt(0));
    expr2tc three = constant_int2tc(uint32, BigInt(3));

    SSA_stepst steps;
    steps.push_back(make_assume(greaterthan2tc(x, three)));
    steps.push_back(make_assert(greaterthan2tc(x, zero)));

    algo.run(steps);

Tha    THEN("Solver finds no counterexample (UNSAT)")
    {
      REQUIRE(algo.solve() == smt_convt::P_UNSATISFIABLE);
    }
  }
}

SCENARIO(
  "incremental_smt_algorithm online mode checks assertions and assumptions immediately",
  "[algorithms][z3][online]")
{
  setup_config();
  contextt ctx;
  namespacet ns(ctx);
  optionst opts = make_options();

  type2tc uint32 = get_uint_type(32);

  GIVEN("Assignment x = 5 then valid assertion x == 5")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x1");
    expr2tc five = constant_int2tc(uint32, BigInt(5));

    algo.init();

    SSA_stept assign = make_assignment(x, five);
    tvt r_assign = algo.step_online(assign);

    THEN("Assignment step returns TV_UNKNOWN")
    {
      REQUIRE(r_assign.is_unknown());
    }

    SSA_stept assrt = make_assert(equality2tc(x, five));
    tvt r_assrt = algo.step_online(assrt);

    THEN("Valid assertion is detected immediately (TV_TRUE)")
    {
      REQUIRE(r_assrt.is_true());
    }
  }

  GIVEN("Assignment x = 5 then violated assertion x == 6")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x2");
    expr2tc five = constant_int2tc(uint32, BigInt(5));
    expr2tc six = constant_int2tc(uint32, BigInt(6));

    algo.init();
    SSA_stept s_assign2 = make_assignment(x, five);
    SSA_stept s_assert2 = make_assert(equality2tc(x, six));
    algo.step_online(s_assign2);
    tvt result = algo.step_online(s_assert2);

    THEN("Violated assertion is detected immediately (TV_FALSE)")
    {
      REQUIRE(result.is_false());
    }
  }

  GIVEN("Assumption x > 3 then valid assertion x > 0")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x3");
    expr2tc zero = constant_int2tc(uint32, BigInt(0));
    expr2tc three = constant_int2tc(uint32, BigInt(3));

    algo.init();
    SSA_stept s_assume3 = make_assume(greaterthan2tc(x, three));
    SSA_stept s_assert3 = make_assert(greaterthan2tc(x, zero));
    algo.step_online(s_assume3);
    tvt result = algo.step_online(s_assert3);

    THEN("Assertion holds under the assumption (TV_TRUE)")
    {
      REQUIRE(result.is_true());
    }
  }

  GIVEN("An ignored step in online mode")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x4");
    SSA_stept s = make_assignment(x, constant_int2tc(uint32, BigInt(7)));
    s.ignore = true;

    algo.init();
    tvt result = algo.step_online(s);

    THEN("Ignored step returns TV_UNKNOWN and is counted")
    {
      REQUIRE(result.is_unknown());
      REQUIRE(algo.ignored() == BigInt(1));
    }
  }

  GIVEN("ask_solver_question called between step_online calls")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x5");
    expr2tc five = constant_int2tc(uint32, BigInt(5));
    expr2tc six = constant_int2tc(uint32, BigInt(6));

    algo.init();
    SSA_stept s_assign5 = make_assignment(x, five);
    algo.step_online(s_assign5);

    THEN("x == 5 is always true at this point")
    {
      REQUIRE(algo.ask_solver_question(equality2tc(x, five)).is_true());
    }
    THEN("x == 6 is always false at this point")
    {
      REQUIRE(algo.ask_solver_question(equality2tc(x, six)).is_false());
    }

    // Solver context is intact after ask_solver_question; the next assert
    // step should still see x = 5.
    SSA_stept s_assert5 = make_assert(equality2tc(x, five));
    tvt result = algo.step_online(s_assert5);
    THEN("Subsequent assertion still holds after ask_solver_question")
    {
      REQUIRE(result.is_true());
    }
  }

  GIVEN("Assumption consistent with current context (x = 5, assume x > 3)")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x6");
    expr2tc five = constant_int2tc(uint32, BigInt(5));
    expr2tc three = constant_int2tc(uint32, BigInt(3));

    algo.init();
    SSA_stept s_assign6 = make_assignment(x, five);
    algo.step_online(s_assign6);

    SSA_stept s_assume6 = make_assume(greaterthan2tc(x, three));
    tvt result = algo.step_online(s_assume6);

    THEN("Assumption guaranteed by current context returns TV_TRUE")
    {
      // x is always 5 and 5 > 3, so the assumption is always true.
      REQUIRE(result.is_true());
    }
  }

  GIVEN("Assumption that contradicts current context (x = 5, assume x > 10)")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x7");
    expr2tc five = constant_int2tc(uint32, BigInt(5));
    expr2tc ten = constant_int2tc(uint32, BigInt(10));

    algo.init();
    SSA_stept s_assign7 = make_assignment(x, five);
    algo.step_online(s_assign7);

    SSA_stept s_assume7 = make_assume(greaterthan2tc(x, ten));
    tvt result = algo.step_online(s_assume7);

    THEN("Infeasible assumption returns TV_FALSE (path is dead)")
    {
      // x is always 5 and 5 > 10 is impossible.
      REQUIRE(result.is_false());
    }
  }

  GIVEN("Unconstrained assumption carries through to later assertion")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x8");
    expr2tc zero = constant_int2tc(uint32, BigInt(0));
    expr2tc four = constant_int2tc(uint32, BigInt(4));

    algo.init();

    // x is unconstrained; assume x > 4 (TV_UNKNOWN at this point).
    SSA_stept s_assume8 = make_assume(greaterthan2tc(x, four));
    tvt r_assume = algo.step_online(s_assume8);

    THEN("Unconstrained assumption returns TV_UNKNOWN")
    {
      REQUIRE(r_assume.is_unknown());
    }

    // x > 4 implies x > 0, so this assertion must hold.
    SSA_stept s_assert8 = make_assert(greaterthan2tc(x, zero));
    tvt r_assert = algo.step_online(s_assert8);

    THEN("Assumption is committed and subsequent assertion sees it (TV_TRUE)")
    {
      REQUIRE(r_assert.is_true());
    }
  }

  GIVEN("Assertion condition is committed to the path like an assumption")
  {
    auto solver = std::unique_ptr<smt_convt>(create_solver("z3", ns, opts));
    incremental_smt_algorithm algo(std::move(solver), false);

    expr2tc x = symbol2tc(uint32, "online::x9");
    expr2tc five = constant_int2tc(uint32, BigInt(5));
    expr2tc six = constant_int2tc(uint32, BigInt(6));

    algo.init();

    // x is unconstrained: asserting x == 5 is undetermined.
    SSA_stept s_assert9a = make_assert(equality2tc(x, five));
    tvt r1 = algo.step_online(s_assert9a);
    THEN("First assert on unconstrained x returns TV_UNKNOWN")
    {
      REQUIRE(r1.is_unknown());
    }

    // The condition x == 5 was committed; x == 5 must now be true.
    SSA_stept s_assert9b = make_assert(equality2tc(x, five));
    tvt r2 = algo.step_online(s_assert9b);
    THEN("Repeated assert on the same condition returns TV_TRUE")
    {
      REQUIRE(r2.is_true());
    }

    // And x == 6 must now be false (x is 5, not 6).
    SSA_stept s_assert9c = make_assert(equality2tc(x, six));
    tvt r3 = algo.step_online(s_assert9c);
    THEN("Contradicting assert returns TV_FALSE")
    {
      REQUIRE(r3.is_false());
    }
  }
}
