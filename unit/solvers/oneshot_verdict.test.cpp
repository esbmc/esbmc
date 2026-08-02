// Contract of oneshot_process::parse_verdict_line, the verdict scanner shared
// by the one-shot subprocess backends (bitwuzllob, neurosym). It must accept
// the bare SMT-LIB verdicts and SAT-competition-style "s ..." lines, tolerate
// surrounding whitespace, map `unknown` to P_ERROR, and reject everything
// else — a scanner that matched substrings would turn a solver's log line
// mentioning "unsat core" into a verification verdict.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <solvers/smtlib/oneshot_process.h>

using oneshot_process::parse_verdict_line;

TEST_CASE("bare SMT-LIB verdicts are recognized", "[oneshot][smtlib]")
{
  REQUIRE(parse_verdict_line("sat") == P_SATISFIABLE);
  REQUIRE(parse_verdict_line("unsat") == P_UNSATISFIABLE);
  REQUIRE(parse_verdict_line("unknown") == P_ERROR);
}

TEST_CASE("SAT-competition-style verdicts are recognized", "[oneshot][smtlib]")
{
  REQUIRE(parse_verdict_line("s SATISFIABLE") == P_SATISFIABLE);
  REQUIRE(parse_verdict_line("s UNSATISFIABLE") == P_UNSATISFIABLE);
  REQUIRE(parse_verdict_line("s UNKNOWN") == P_ERROR);
}

TEST_CASE("surrounding whitespace is tolerated", "[oneshot][smtlib]")
{
  REQUIRE(parse_verdict_line("  sat") == P_SATISFIABLE);
  REQUIRE(parse_verdict_line("unsat\r\n") == P_UNSATISFIABLE);
  REQUIRE(parse_verdict_line("\t s SATISFIABLE \n") == P_SATISFIABLE);
}

TEST_CASE("non-verdict lines are rejected", "[oneshot][smtlib]")
{
  REQUIRE_FALSE(parse_verdict_line("").has_value());
  REQUIRE_FALSE(parse_verdict_line("   ").has_value());
  REQUIRE_FALSE(parse_verdict_line("c solving /tmp/q.smt2").has_value());
  REQUIRE_FALSE(parse_verdict_line("[NeuroSym] loading model...").has_value());
  REQUIRE_FALSE(parse_verdict_line("unsat core").has_value());
  REQUIRE_FALSE(parse_verdict_line("presat").has_value());
  REQUIRE_FALSE(parse_verdict_line("SATISFIABLE").has_value());
  REQUIRE_FALSE(parse_verdict_line("s sat").has_value());
}
