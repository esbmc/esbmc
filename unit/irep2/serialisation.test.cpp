// R7 / P3 serialisation: every enum value handled by a type_to_string overload
// must map to a printable name (no fall-through to the assert(0 &&
// "Unrecognized…") tail), and every expr/type kind must have a non-empty
// pretty name. An out-of-range enum value must hit the deliberate abort. All
// checks run against the real irep2 stringifiers.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <functional>
#include <string>
#include <vector>

// The out-of-range abort death-test forks and waits on a POSIX child; guard the
// mechanism (and its test case) out on Windows, which has neither fork() nor
// <sys/wait.h>.
#if !defined(_WIN32)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <irep2/irep2.h>
#include <irep2/irep2_dispatch.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
#if !defined(_WIN32)
void require_aborts(const std::function<void()> &fn)
{
  pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0)
  {
    // Consume the freopen result to satisfy -Werror=unused-result.
    if (!freopen("/dev/null", "w", stderr) || !freopen("/dev/null", "w", stdout))
      _exit(2);
    fn();
    _exit(0); // unreachable if fn aborts
  }
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFSIGNALED(status));
  REQUIRE(WTERMSIG(status) == SIGABRT);
}
#endif
} // namespace

TEST_CASE("type_to_string names every enum value (R7)", "[core][irep2]")
{
  using csk = constant_string_kindt;
  for (csk v : {csk::DEFAULT, csk::WIDE, csk::UNICODE})
    REQUIRE(!type_to_string(v, 0).empty());

  using srl = symbol_renaming_level;
  for (srl v :
       {srl::level0,
        srl::level1,
        srl::level2,
        srl::level1_global,
        srl::level2_global})
    REQUIRE(!type_to_string(v, 0).empty());

  using pk = printf_kindt;
  for (pk v :
       {pk::PRINTF,
        pk::FPRINTF,
        pk::DPRINTF,
        pk::SPRINTF,
        pk::VFPRINTF,
        pk::SNPRINTF,
        pk::VPRINTF,
        pk::VSPRINTF,
        pk::VSNPRINTF,
        pk::ASPRINTF,
        pk::VASPRINTF})
    REQUIRE(!type_to_string(v, 0).empty());

  using sak = sideeffect_allockind;
  for (sak v : {sak::malloc,           sak::realloc,
                sak::alloca,           sak::cpp_new,
                sak::cpp_new_arr,      sak::nondet,
                sak::va_arg,           sak::printf2,
                sak::function_call,    sak::preincrement,
                sak::postincrement,    sak::predecrement,
                sak::postdecrement,    sak::old_snapshot,
                sak::assigns_target,   sak::statement_expression,
                sak::temporary_object, sak::gcc_conditional_expression,
                sak::cpp_delete,       sak::cpp_delete_array})
    REQUIRE(!type_to_string(v, 0).empty());
}

TEST_CASE("every kind has a non-empty pretty name (R7)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  type2tc u32 = get_uint_type(32);
  expr2tc c5 = constant_int2tc(u32, BigInt(5));

  std::vector<type2tc> types{
    get_bool_type(),
    u32,
    get_int_type(64),
    pointer_type2tc(u32),
    array_type2tc(u32, constant_int2tc(u32, BigInt(2)), false),
  };
  for (const type2tc &t : types)
  {
    REQUIRE(!get_type_id(t).empty());
    REQUIRE(!t->pretty(0).empty());
  }

  std::vector<expr2tc> exprs{
    c5,
    constant_bool2tc(true),
    symbol2tc(u32, "x"),
    add2tc(u32, c5, c5),
    not2tc(constant_bool2tc(true)),
  };
  for (const expr2tc &e : exprs)
  {
    REQUIRE(!get_expr_id(e).empty());
    REQUIRE(!e->pretty(0).empty());
  }
}

#if !defined(_WIN32)
TEST_CASE("type_to_string aborts on an out-of-range enum (R7)", "[core][irep2]")
{
  // A value past the last enumerator falls through the switch to the
  // assert(0 && "Unrecognized…") tail.
  require_aborts(
    []() { (void)type_to_string(static_cast<constant_string_kindt>(99), 0); });
}
#endif
